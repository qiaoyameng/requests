"""Tests for HTTP Digest authentication with qop=auth-int.

These tests run against a minimal local server, implemented in-process
behind a custom transport adapter, which independently recomputes the
expected digest response from the exact entity bytes it receives.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from urllib.parse import urlparse

import pytest
from urllib3 import HTTPResponse

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth

_ALGORITHMS = {"MD5": "md5", "SHA-256": "sha256"}

_AUTH_PARAM_RE = re.compile(r'(\w[\w-]*)=(?:"((?:[^"\\]|\\.)*)"|([^,\s]+))')


def _read_entity(body):
    """Read a prepared body exactly as it would be sent on the wire.

    Transfer-encoding boundaries never appear here, and streams are left
    fully consumed, as they would be after a real transmit.
    """
    if body is None:
        return b""
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, (bytes, bytearray, memoryview)):
        return bytes(body)
    read = getattr(body, "read", None)
    if read is not None:
        chunks = []
        while True:
            chunk = read(65536)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            chunks.append(chunk)
        return b"".join(chunks)
    return b"".join(
        chunk.encode("utf-8") if isinstance(chunk, str) else chunk for chunk in body
    )


def _parse_digest_params(value):
    scheme, _, rest = value.partition(" ")
    params = {}
    for match in _AUTH_PARAM_RE.finditer(rest):
        key, quoted, bare = match.groups()
        params[key] = quoted if quoted is not None else bare
    return scheme, params


class LocalServerAdapter(HTTPAdapter):
    """Routes requests to an in-process handler instead of the network."""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def send(self, request, **kwargs):
        body = _read_entity(request.body)
        status, reason, headers, content = self.handler(request, body)
        raw = HTTPResponse(
            body=io.BytesIO(content),
            status=status,
            reason=reason,
            headers=headers,
            preload_content=False,
        )
        return self.build_response(request, raw)


class DigestAuthIntServer:
    """Minimal local digest-auth server.

    The expected digest response is recomputed independently from the
    exact entity bytes received, so the client's entity hashing is
    verified end to end.
    """

    def __init__(
        self,
        username="user",
        password="pass",
        realm="testrealm",
        qop="auth-int",
        expect_qop=None,
        algorithm="MD5",
        stale_once=False,
        on_success=None,
    ):
        self.username = username
        self.password = password
        self.realm = realm
        self.qop = qop
        self.expect_qop = expect_qop or qop
        self.algorithm = algorithm
        self.stale_once = stale_once
        self.on_success = on_success
        self._stale_sent = False
        self.nonce = os.urandom(8).hex()
        self.received = []

    def _hash(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.new(_ALGORITHMS[self.algorithm])
        digest.update(data)
        return digest.hexdigest()

    def _expected_response(self, params, method, body):
        try:
            ha1 = self._hash(f"{self.username}:{self.realm}:{self.password}")
            a2 = f"{method}:{params['uri']}"
            if self.expect_qop == "auth-int":
                a2 = f"{a2}:{self._hash(body)}"
            ha2 = self._hash(a2)
            noncebit = (
                f"{params['nonce']}:{params['nc']}:{params['cnonce']}"
                f":{self.expect_qop}:{ha2}"
            )
            return self._hash(f"{ha1}:{noncebit}")
        except KeyError:
            return None

    def _challenge(self, stale=False):
        parts = [
            f'realm="{self.realm}"',
            f'nonce="{self.nonce}"',
            f'qop="{self.qop}"',
            f"algorithm={self.algorithm}",
        ]
        if stale:
            parts.append("stale=true")
        return "Digest " + ", ".join(parts)

    def handle(self, request, body):
        parsed = urlparse(request.url)
        uri = parsed.path or "/"
        if parsed.query:
            uri += f"?{parsed.query}"
        scheme, params = _parse_digest_params(request.headers.get("Authorization", ""))
        self.received.append(
            {
                "method": request.method,
                "uri": uri,
                "headers": request.headers,
                "body": body,
                "params": params,
            }
        )

        valid = (
            scheme.lower() == "digest"
            and params.get("username") == self.username
            and params.get("nonce") == self.nonce
            and params.get("uri") == uri
            and params.get("qop") == self.expect_qop
            and params.get("response")
            == self._expected_response(params, request.method, body)
        )

        if valid and self.stale_once and not self._stale_sent:
            # Force the client to retry once with a fresh nonce.
            self._stale_sent = True
            self.nonce = os.urandom(8).hex()
            return (
                401,
                "Unauthorized",
                {"WWW-Authenticate": self._challenge(stale=True)},
                b"",
            )
        if valid:
            if self.on_success is not None:
                return self.on_success()
            return 200, "OK", {}, b"authenticated"
        return 401, "Unauthorized", {"WWW-Authenticate": self._challenge()}, b""


def digest_session(server, host="digest.test"):
    session = requests.Session()
    session.trust_env = False
    session.mount(f"http://{host}/", LocalServerAdapter(server.handle))
    return session


def test_digest_auth_int_binary_body():
    server = DigestAuthIntServer()
    body = bytes(range(256)) * 4
    session = digest_session(server)
    auth = HTTPDigestAuth("user", "pass")

    r = session.post("http://digest.test/submit?x=1", data=body, auth=auth)
    assert r.status_code == 200
    assert len(r.history) == 1
    authed = server.received[1]
    assert authed["body"] == body
    assert authed["params"]["qop"] == "auth-int"

    # A second request reuses the saved nonce and authenticates
    # preemptively, hashing the new entity in the process.
    other = b"\x00\xff" * 128
    r = session.post("http://digest.test/submit", data=other, auth=auth)
    assert r.status_code == 200
    assert len(server.received) == 3
    assert server.received[2]["body"] == other
    assert server.received[2]["params"]["nc"] == "00000002"


def test_digest_auth_int_get_without_body():
    server = DigestAuthIntServer()
    session = digest_session(server)
    r = session.get("http://digest.test/resource", auth=HTTPDigestAuth("user", "pass"))
    assert r.status_code == 200
    assert server.received[1]["body"] == b""


def test_digest_auth_int_stream_with_nonzero_offset(tmp_path):
    content = os.urandom(4096)
    offset = 517
    (tmp_path / "body.bin").write_bytes(content)
    server = DigestAuthIntServer()
    session = digest_session(server)
    with open(tmp_path / "body.bin", "rb") as body_file:
        body_file.seek(offset)
        r = session.post(
            "http://digest.test/upload",
            data=body_file,
            auth=HTTPDigestAuth("user", "pass"),
        )
        assert r.status_code == 200
        # The entity digest (and the wire body) covers the bytes from the
        # initial stream position onwards.
        assert server.received[1]["body"] == content[offset:]
        # Sending consumed the body to the end of the stream.
        assert body_file.tell() == len(content)


def test_digest_auth_int_stale_retry():
    server = DigestAuthIntServer(stale_once=True)
    session = digest_session(server)
    r = session.post(
        "http://digest.test/stale",
        data=b"stale-body",
        auth=HTTPDigestAuth("user", "pass"),
    )
    assert r.status_code == 200
    assert len(r.history) == 2
    assert len(server.received) == 3
    first, second = server.received[1:]
    # The stale challenge supplied a new nonce and the nonce count reset.
    assert first["params"]["nonce"] != second["params"]["nonce"]
    assert first["params"]["nc"] == second["params"]["nc"] == "00000001"


def test_digest_auth_int_sha256():
    server = DigestAuthIntServer(algorithm="SHA-256")
    session = digest_session(server)
    r = session.post(
        "http://digest.test/sha",
        data=b"sha-256 body",
        auth=HTTPDigestAuth("user", "pass"),
    )
    assert r.status_code == 200


def test_digest_auth_qop_auth_still_supported():
    server = DigestAuthIntServer(qop="auth")
    session = digest_session(server)
    r = session.post(
        "http://digest.test/auth",
        data=b"auth mode body",
        auth=HTTPDigestAuth("user", "pass"),
    )
    assert r.status_code == 200
    assert server.received[1]["params"]["qop"] == "auth"


def test_digest_auth_prefers_auth_when_both_offered():
    server = DigestAuthIntServer(qop="auth,auth-int", expect_qop="auth")
    session = digest_session(server)
    r = session.post(
        "http://digest.test/either",
        data=b"either body",
        auth=HTTPDigestAuth("user", "pass"),
    )
    assert r.status_code == 200
    assert server.received[1]["params"]["qop"] == "auth"


def test_digest_auth_int_rejects_non_replayable_body():
    server = DigestAuthIntServer()
    session = digest_session(server)

    def gen():
        yield b"chunk-1"
        yield b"chunk-2"

    with pytest.raises(ValueError, match="auth-int"):
        session.post(
            "http://digest.test/stream",
            data=gen(),
            auth=HTTPDigestAuth("user", "pass"),
        )
    # Only the unauthenticated request was sent; the retry was refused.
    assert len(server.received) == 1

    # Once a nonce is known, the rejection happens before anything is sent.
    auth = HTTPDigestAuth("user", "pass")
    r = session.post("http://digest.test/ok", data=b"fine", auth=auth)
    assert r.status_code == 200
    with pytest.raises(ValueError, match="auth-int"):
        session.post("http://digest.test/stream", data=gen(), auth=auth)
    assert len(server.received) == 3


def test_digest_auth_int_redirect_does_not_leak_authorization():
    received_b = []

    def handle_b(request, body):
        received_b.append(request.headers)
        return 200, "OK", {}, b"cross-origin"

    server_a = DigestAuthIntServer(
        on_success=lambda: (302, "Found", {"Location": "http://b.test/landing"}, b"")
    )
    session = digest_session(server_a, host="a.test")
    session.mount("http://b.test/", LocalServerAdapter(handle_b))

    r = session.post(
        "http://a.test/start", data=b"redirect-me", auth=HTTPDigestAuth("user", "pass")
    )
    assert r.status_code == 200
    # The authenticated request succeeded against the first origin...
    assert server_a.received[1]["params"].get("response")
    # ...but the redirected request carried no credentials.
    assert len(received_b) == 1
    assert "Authorization" not in received_b[0]
