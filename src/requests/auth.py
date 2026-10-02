"""
requests.auth
~~~~~~~~~~~~~

This module contains the authentication handlers for Requests.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import warnings
from base64 import b64encode
from typing import TYPE_CHECKING, Any, Callable, Final, cast, overload

from ._internal_utils import to_native_string
from .compat import basestring, str, urlparse
from .cookies import extract_cookies_to_jar
from .utils import parse_dict_header

if TYPE_CHECKING:
    from http.cookiejar import CookieJar
    from typing import Any

    from .models import PreparedRequest, Response

CONTENT_TYPE_FORM_URLENCODED: Final = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART: Final = "multipart/form-data"


def _get_entity_body_digest(body: Any, hash_constructor: Callable[..., Any]) -> str:
    """Return the hex digest of the exact bytes of a request entity body.

    The digest covers the entity as it will be sent on the wire: transfer
    encoding boundaries (e.g. chunks) are never mixed in, and text is
    encoded to UTF-8 exactly once. Seekable streams are hashed from their
    current position, which is restored afterwards so the body can still
    be transmitted. Bodies that cannot be replayed (e.g. generators or
    non-seekable streams) are rejected outright instead of being silently
    treated as empty.
    """
    if body is None:
        return hash_constructor(b"", usedforsecurity=False).hexdigest()
    if isinstance(body, str):
        body = body.encode("utf-8")
    if isinstance(body, (bytes, bytearray, memoryview)):
        return hash_constructor(body, usedforsecurity=False).hexdigest()

    read = getattr(body, "read", None)
    tell = getattr(body, "tell", None)
    seek = getattr(body, "seek", None)
    if read is None or tell is None or seek is None:
        raise ValueError(
            "Cannot compute the entity digest required by qop=auth-int: "
            "the request body is not replayable. Digest authentication "
            "with qop=auth-int requires a body of bytes or a seekable "
            "binary stream."
        )
    try:
        pos = tell()
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Cannot compute the entity digest required by qop=auth-int: "
            "the position of the request body stream cannot be determined."
        ) from exc

    digest = hash_constructor(usedforsecurity=False)
    try:
        while True:
            chunk = read(65536)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            digest.update(chunk)
    finally:
        # Restore the original position so the body can still be sent.
        seek(pos)

    return digest.hexdigest()


def _basic_auth_str(username: bytes | str, password: bytes | str) -> str:
    """Returns a Basic Auth string."""

    # "I want us to put a big-ol' comment on top of it that
    # says that this behaviour is dumb but we need to preserve
    # it because people are relying on it."
    #    - Lukasa
    #
    # These are here solely to maintain backwards compatibility
    # for things like ints. This will be removed in 3.0.0.
    if not isinstance(username, basestring):  # runtime guard for non-str/bytes
        warnings.warn(
            "Non-string usernames will no longer be supported in Requests "
            f"3.0.0. Please convert the object you've passed in ({username!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.",
            category=DeprecationWarning,
        )
        username = str(username)

    if not isinstance(password, basestring):  # runtime guard for non-str/bytes
        warnings.warn(
            "Non-string passwords will no longer be supported in Requests "
            f"3.0.0. Please convert the object you've passed in ({type(password)!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.",
            category=DeprecationWarning,
        )
        password = str(password)
    # -- End Removal --

    if isinstance(username, str):
        username = username.encode("latin1")

    if isinstance(password, str):
        password = password.encode("latin1")

    authstr = "Basic " + to_native_string(
        b64encode(b":".join((username, password))).strip()
    )

    return authstr


class AuthBase:
    """Base class that all auth implementations derive from"""

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        raise NotImplementedError("Auth hooks must be callable.")


class HTTPBasicAuth(AuthBase):
    """Attaches HTTP Basic Authentication to the given Request object."""

    username: bytes | str
    password: bytes | str

    @overload
    def __init__(self, username: str, password: str) -> None: ...
    @overload
    def __init__(self, username: bytes, password: bytes) -> None: ...

    def __init__(self, username: bytes | str, password: bytes | str) -> None:
        self.username = username
        self.password = password

    def __eq__(self, other: object) -> bool:
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers["Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPProxyAuth(HTTPBasicAuth):
    """Attaches HTTP Proxy Authentication to a given Request object."""

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers["Proxy-Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPDigestAuth(AuthBase):
    """Attaches HTTP Digest Authentication to the given Request object."""

    username: bytes | str
    password: bytes | str
    _thread_local: threading.local
    last_nonce: str
    nonce_count: int
    chal: dict[str, str]
    pos: int | None
    num_401_calls: int | None

    @overload
    def __init__(self, username: str, password: str) -> None: ...
    @overload
    def __init__(self, username: bytes, password: bytes) -> None: ...

    def __init__(self, username: bytes | str, password: bytes | str) -> None:
        self.username = username
        self.password = password
        # Keep state in per-thread local storage
        self._thread_local = threading.local()

    def init_per_thread_state(self) -> None:
        # Ensure state is initialized just once per-thread
        if not hasattr(self._thread_local, "init"):
            self._thread_local.init = True
            self._thread_local.last_nonce = ""
            self._thread_local.nonce_count = 0
            self._thread_local.chal = {}
            self._thread_local.pos = None
            self._thread_local.num_401_calls = None

    def build_digest_header(
        self, method: str, url: str, body: Any = None
    ) -> str | None:
        """
        :rtype: str
        """

        realm = self._thread_local.chal["realm"]
        nonce = self._thread_local.chal["nonce"]
        qop = self._thread_local.chal.get("qop")
        algorithm = self._thread_local.chal.get("algorithm")
        opaque = self._thread_local.chal.get("opaque")

        if algorithm is None:
            _algorithm = "MD5"
        else:
            _algorithm = algorithm.upper()
        # constructors assume digest modules are imported at the top level
        if _algorithm == "MD5" or _algorithm == "MD5-SESS":
            hash_constructor = hashlib.md5
        elif _algorithm == "SHA":
            hash_constructor = hashlib.sha1
        elif _algorithm == "SHA-256":
            hash_constructor = hashlib.sha256
        elif _algorithm == "SHA-512":
            hash_constructor = hashlib.sha512
        else:
            # Unsupported algorithm: cannot build a digest header.
            return None

        def hash_utf8(x: str | bytes) -> str:
            if isinstance(x, str):
                x = x.encode("utf-8")
            return hash_constructor(x, usedforsecurity=False).hexdigest()

        def KD(s: str, d: str) -> str:
            return hash_utf8(f"{s}:{d}")

        p_parsed = urlparse(url)
        #: path is request-uri defined in RFC 2616 which should not be empty
        path = p_parsed.path or "/"
        if p_parsed.query:
            path += f"?{p_parsed.query}"

        A1 = f"{self.username}:{realm}:{self.password}"
        A2 = f"{method}:{path}"

        HA1 = hash_utf8(A1)

        if nonce == self._thread_local.last_nonce:
            self._thread_local.nonce_count += 1
        else:
            self._thread_local.nonce_count = 1
        ncvalue = f"{self._thread_local.nonce_count:08x}"
        s = str(self._thread_local.nonce_count).encode("utf-8")
        s += nonce.encode("utf-8")
        s += time.ctime().encode("utf-8")
        s += os.urandom(8)

        cnonce = hashlib.sha1(s, usedforsecurity=False).hexdigest()[:16]
        if _algorithm == "MD5-SESS":
            HA1 = hash_utf8(f"{HA1}:{nonce}:{cnonce}")  # type: ignore[reportConstantRedefinition]  # RFC 2617 terminology

        qop_options = [option.strip() for option in qop.split(",")] if qop else []
        qop_used: str | None
        if "auth" in qop_options:
            # Prefer "auth" when the server offers a choice: it does not
            # require hashing the request entity.
            qop_used = "auth"
        elif "auth-int" in qop_options:
            qop_used = "auth-int"
        elif qop_options:
            # The server only offered qop options we don't support.
            return None
        else:
            qop_used = None

        if qop_used == "auth-int":
            # RFC 7616: for auth-int, A2 additionally contains the hash of
            # the exact bytes of the request entity.
            entdig = _get_entity_body_digest(body, hash_constructor)
            A2 = f"{A2}:{entdig}"

        HA2 = hash_utf8(A2)

        if qop_used is None:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        else:
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:{qop_used}:{HA2}"
            respdig = KD(HA1, noncebit)

        self._thread_local.last_nonce = nonce

        # XXX should the partial digests be encoded too?
        base = (
            f'username="{self.username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{path}", response="{respdig}"'
        )
        if opaque:
            base += f', opaque="{opaque}"'
        if algorithm:
            base += f', algorithm="{algorithm}"'
        if qop_used:
            base += f', qop="{qop_used}", nc={ncvalue}, cnonce="{cnonce}"'

        return f"Digest {base}"

    def handle_redirect(self, r: Response, **kwargs: Any) -> None:
        """Reset num_401_calls counter on redirects."""
        if r.is_redirect:
            self._thread_local.num_401_calls = 1

    def handle_401(self, r: Response, **kwargs: Any) -> Response:
        """
        Takes the given response and tries digest-auth, if needed.

        :rtype: requests.Response
        """

        # If response is not 4xx, do not auth
        # See https://github.com/psf/requests/issues/3772
        if not 400 <= r.status_code < 500:
            self._thread_local.num_401_calls = 1
            return r

        # A single request may be challenged more than once (e.g. when the
        # server replies with a stale nonce), so keep answering challenges
        # until the server accepts the response or stops challenging us.
        while True:
            if self._thread_local.pos is not None:
                # Rewind the file position indicator of the body to where
                # it was to resend the request.
                if (seek := getattr(r.request.body, "seek", None)) is not None:
                    seek(self._thread_local.pos)
            s_auth = r.headers.get("www-authenticate", "")

            if "digest" not in s_auth.lower():
                self._thread_local.num_401_calls = 1
                return r

            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            chal = parse_dict_header(pat.sub("", s_auth, count=1))

            # A stale challenge means the server rejected the nonce (e.g. it
            # expired) but not necessarily the credentials, so allow one
            # extra attempt to retry with the fresh nonce it provided.
            stale = (chal.get("stale") or "").lower() == "true"
            max_401_calls = 3 if stale else 2
            if self._thread_local.num_401_calls >= max_401_calls:
                self._thread_local.num_401_calls = 1
                return r

            self._thread_local.num_401_calls += 1
            self._thread_local.chal = chal

            # Consume content and release the original connection
            # to allow our new request to reuse the same one.
            r.content
            r.close()
            prep = r.request.copy()
            cookie_jar = cast("CookieJar", prep._cookies)
            extract_cookies_to_jar(cookie_jar, r.request, r.raw)
            prep.prepare_cookies(cookie_jar)

            _digest_auth = self.build_digest_header(
                cast(str, prep.method), cast(str, prep.url), prep.body
            )
            if _digest_auth:
                prep.headers["Authorization"] = _digest_auth
            _r = r.connection.send(prep, **kwargs)
            _r.history.extend(r.history)
            _r.history.append(r)
            _r.request = prep

            r = _r
            if not 400 <= r.status_code < 500:
                self._thread_local.num_401_calls = 1
                return r

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        # Initialize per-thread state, if needed
        self.init_per_thread_state()
        # If we have a saved nonce, skip the 401
        if self._thread_local.last_nonce:
            _digest_auth = self.build_digest_header(
                cast(str, r.method), cast(str, r.url), r.body
            )
            if _digest_auth:
                r.headers["Authorization"] = _digest_auth
        if (tell := getattr(r.body, "tell", None)) is not None:
            self._thread_local.pos = tell()
        else:
            # In the case of HTTPDigestAuth being reused and the body of
            # the previous request was a file-like object, pos has the
            # file position of the previous body. Ensure it's set to
            # None.
            self._thread_local.pos = None
        r.register_hook("response", self.handle_401)
        r.register_hook("response", self.handle_redirect)
        self._thread_local.num_401_calls = 1

        return r

    def __eq__(self, other: object) -> bool:
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other
