import io
import threading

import pytest
from urllib3 import HTTPResponse

import requests
from requests.adapters import HTTPAdapter
from requests.compat import JSONDecodeError
from tests.testserver.server import Server, consume_socket_content

from .utils import override_environ


def echo_response_handler(sock):
    """Simple handler that will take request and echo it back to requester."""
    request_content = consume_socket_content(sock, timeout=0.5)

    text_200 = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: %d\r\n\r\n"
        b"%s"
    ) % (len(request_content), request_content)
    sock.send(text_200)


def test_chunked_upload():
    """can safely send generators"""
    close_server = threading.Event()
    server = Server.basic_response_server(wait_to_close_event=close_server)
    data = iter([b"a", b"b", b"c"])

    with server as (host, port):
        url = f"http://{host}:{port}/"
        r = requests.post(url, data=data, stream=True)
        close_server.set()  # release server block

    assert r.status_code == 200
    assert r.request.headers["Transfer-Encoding"] == "chunked"


def test_chunked_encoding_error():
    """get a ChunkedEncodingError if the server returns a bad response"""

    def incomplete_chunked_response_handler(sock):
        request_content = consume_socket_content(sock, timeout=0.5)

        # The server never ends the request and doesn't provide any valid chunks
        sock.send(
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
        )

        return request_content

    close_server = threading.Event()
    server = Server(incomplete_chunked_response_handler)

    with server as (host, port):
        url = f"http://{host}:{port}/"
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            requests.get(url)
        close_server.set()  # release server block


def test_chunked_upload_uses_only_specified_host_header():
    """Ensure we use only the specified Host header for chunked requests."""
    close_server = threading.Event()
    server = Server(echo_response_handler, wait_to_close_event=close_server)

    data = iter([b"a", b"b", b"c"])
    custom_host = "sample-host"

    with server as (host, port):
        url = f"http://{host}:{port}/"
        r = requests.post(url, data=data, headers={"Host": custom_host}, stream=True)
        close_server.set()  # release server block

    expected_header = b"Host: %s\r\n" % custom_host.encode("utf-8")
    assert expected_header in r.content
    assert r.content.count(b"Host: ") == 1


def test_chunked_upload_doesnt_skip_host_header():
    """Ensure we don't omit all Host headers with chunked requests."""
    close_server = threading.Event()
    server = Server(echo_response_handler, wait_to_close_event=close_server)

    data = iter([b"a", b"b", b"c"])

    with server as (host, port):
        expected_host = f"{host}:{port}"
        url = f"http://{host}:{port}/"
        r = requests.post(url, data=data, stream=True)
        close_server.set()  # release server block

    expected_header = b"Host: %s\r\n" % expected_host.encode("utf-8")
    assert expected_header in r.content
    assert r.content.count(b"Host: ") == 1


def test_conflicting_content_lengths():
    """Ensure we correctly throw an InvalidHeader error if multiple
    conflicting Content-Length headers are returned.
    """

    def multiple_content_length_response_handler(sock):
        request_content = consume_socket_content(sock, timeout=0.5)
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 16\r\n"
            b"Content-Length: 32\r\n\r\n"
            b"-- Bad Actor -- Original Content\r\n"
        )
        sock.send(response)

        return request_content

    close_server = threading.Event()
    server = Server(multiple_content_length_response_handler)

    with server as (host, port):
        url = f"http://{host}:{port}/"
        with pytest.raises(requests.exceptions.InvalidHeader):
            requests.get(url)
        close_server.set()


def test_digestauth_401_count_reset_on_redirect():
    """Ensure we correctly reset num_401_calls after a successful digest auth,
    followed by a 302 redirect to another digest auth prompt.

    See https://github.com/psf/requests/issues/1979.
    """
    text_401 = (b'HTTP/1.1 401 UNAUTHORIZED\r\n'
                b'Content-Length: 0\r\n'
                b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                b', opaque="372825293d1c26955496c80ed6426e9e", '
                b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    text_302 = (b'HTTP/1.1 302 FOUND\r\n'
                b'Content-Length: 0\r\n'
                b'Location: /\r\n\r\n')

    text_200 = (b'HTTP/1.1 200 OK\r\n'
                b'Content-Length: 0\r\n\r\n')

    expected_digest = (b'Authorization: Digest username="user", '
                       b'realm="me@kennethreitz.com", '
                       b'nonce="6bf5d6e4da1ce66918800195d6b9130d", uri="/"')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to initial GET with a challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_401)

        # Verify we receive an Authorization header in response, then redirect.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_302)

        # Verify Authorization isn't sent to the redirected host,
        # then send another challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert b'Authorization:' not in request_content
        sock.send(text_401)

        # Verify Authorization is sent correctly again, and return 200 OK.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_200)

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url, auth=auth)
        # Verify server succeeded in authenticating.
        assert r.status_code == 200
        # Verify Authorization was sent in final request.
        assert 'Authorization' in r.request.headers
        assert r.request.headers['Authorization'].startswith('Digest ')
        # Verify redirect happened as we expected.
        assert r.history[0].status_code == 302
        close_server.set()


def test_digestauth_401_only_sent_once():
    """Ensure we correctly respond to a 401 challenge once, and then
    stop responding if challenged again.
    """
    text_401 = (b'HTTP/1.1 401 UNAUTHORIZED\r\n'
                b'Content-Length: 0\r\n'
                b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                b', opaque="372825293d1c26955496c80ed6426e9e", '
                b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    expected_digest = (b'Authorization: Digest username="user", '
                       b'realm="me@kennethreitz.com", '
                       b'nonce="6bf5d6e4da1ce66918800195d6b9130d", uri="/"')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_failed_response_handler(sock):
        # Respond to initial GET with a challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_401)

        # Verify we receive an Authorization header in response, then
        # challenge again.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_401)

        # Verify the client didn't respond to second challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_failed_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url, auth=auth)
        # Verify server didn't authenticate us.
        assert r.status_code == 401
        assert r.history[0].status_code == 401
        close_server.set()


def test_digestauth_only_on_4xx():
    """Ensure we only send digestauth on 4xx challenges.

    See https://github.com/psf/requests/issues/3772.
    """
    text_200_chal = (b'HTTP/1.1 200 OK\r\n'
                     b'Content-Length: 0\r\n'
                     b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                     b', opaque="372825293d1c26955496c80ed6426e9e", '
                     b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to GET with a 200 containing www-authenticate header.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_200_chal)

        # Verify the client didn't respond with auth.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url, auth=auth)
        # Verify server didn't receive auth from us.
        assert r.status_code == 200
        assert len(r.history) == 0
        close_server.set()


_schemes_by_var_prefix = [
    ('http', ['http']),
    ('https', ['https']),
    ('all', ['http', 'https']),
]

_proxy_combos = []
for prefix, schemes in _schemes_by_var_prefix:
    for scheme in schemes:
        _proxy_combos.append((f"{prefix}_proxy", scheme))

_proxy_combos += [(var.upper(), scheme) for var, scheme in _proxy_combos]


@pytest.mark.parametrize("var,scheme", _proxy_combos)
def test_use_proxy_from_environment(httpbin, var, scheme):
    url = f"{scheme}://httpbin.org"
    fake_proxy = Server()  # do nothing with the requests; just close the socket
    with fake_proxy as (host, port):
        proxy_url = f"socks5://{host}:{port}"
        kwargs = {var: proxy_url}
        with override_environ(**kwargs):
            # fake proxy's lack of response will cause a ConnectionError
            with pytest.raises(requests.exceptions.ConnectionError):
                requests.get(url)

        # the fake proxy received a request
        assert len(fake_proxy.handler_results) == 1

        # it had actual content (not checking for SOCKS protocol for now)
        assert len(fake_proxy.handler_results[0]) > 0


def test_redirect_rfc1808_to_non_ascii_location():
    path = 'š'
    expected_path = b'%C5%A1'
    redirect_request = []  # stores the second request to the server

    def redirect_resp_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        location = f'//{host}:{port}/{path}'
        sock.send(
            (
                b'HTTP/1.1 301 Moved Permanently\r\n'
                b'Content-Length: 0\r\n'
                b'Location: %s\r\n'
                b'\r\n'
            ) % location.encode('utf8')
        )
        redirect_request.append(consume_socket_content(sock, timeout=0.5))
        sock.send(b'HTTP/1.1 200 OK\r\n\r\n')

    close_server = threading.Event()
    server = Server(redirect_resp_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}'
        r = requests.get(url=url, allow_redirects=True)
        assert r.status_code == 200
        assert len(r.history) == 1
        assert r.history[0].status_code == 301
        assert redirect_request[0].startswith(b'GET /' + expected_path + b' HTTP/1.1')
        assert r.url == '{}/{}'.format(url, expected_path.decode('ascii'))

        close_server.set()


def test_fragment_not_sent_with_request():
    """Verify that the fragment portion of a URI isn't sent to the server."""
    close_server = threading.Event()
    server = Server(echo_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/path/to/thing/#view=edit&token=hunter2'
        r = requests.get(url)
        raw_request = r.content

        assert r.status_code == 200
        headers, body = raw_request.split(b'\r\n\r\n', 1)
        status_line, headers = headers.split(b'\r\n', 1)

        assert status_line == b'GET /path/to/thing/ HTTP/1.1'
        for frag in (b'view', b'edit', b'token', b'hunter2'):
            assert frag not in headers
            assert frag not in body

        close_server.set()


def test_fragment_update_on_redirect():
    """Verify we only append previous fragment if one doesn't exist on new
    location. If a new fragment is encountered in a Location header, it should
    be added to all subsequent requests.
    """

    def response_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 302 FOUND\r\n'
            b'Content-Length: 0\r\n'
            b'Location: /get#relevant-section\r\n\r\n'
        )
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 302 FOUND\r\n'
            b'Content-Length: 0\r\n'
            b'Location: /final-url/\r\n\r\n'
        )
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 200 OK\r\n\r\n'
        )

    close_server = threading.Event()
    server = Server(response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/path/to/thing/#view=edit&token=hunter2'
        r = requests.get(url)

        assert r.status_code == 200
        assert len(r.history) == 2
        assert r.history[0].request.url == url

        # Verify we haven't overwritten the location with our previous fragment.
        assert r.history[1].request.url == f'http://{host}:{port}/get#relevant-section'
        # Verify previous fragment is used and not the original.
        assert r.url == f'http://{host}:{port}/final-url/#relevant-section'

        close_server.set()


def test_json_decode_compatibility_for_alt_utf_encodings():

    def response_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 200 OK\r\n'
            b'Content-Length: 18\r\n\r\n'
            b'\xff\xfe{\x00"\x00K0"\x00=\x00"\x00\xab0"\x00\r\n'
        )

    close_server = threading.Event()
    server = Server(response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url)
    r.encoding = None
    with pytest.raises(requests.exceptions.JSONDecodeError) as excinfo:
        r.json()
    assert isinstance(excinfo.value, requests.exceptions.RequestException)
    assert isinstance(excinfo.value, JSONDecodeError)
    assert r.text not in str(excinfo.value)


class CountingStream:
    """A non-seekable readable stream (like a temporary decompression
    stream) that counts how many times Requests closes it."""

    def __init__(self, payload, on_close, fail_after_reads=None):
        self._payload = payload
        self._position = 0
        self._on_close = on_close
        self._fail_after_reads = fail_after_reads
        self._reads = 0

    def read(self, size=-1):
        self._reads += 1
        if self._fail_after_reads is not None and self._reads > self._fail_after_reads:
            raise RuntimeError("stream broke mid-send")
        if self._position >= len(self._payload):
            return b""
        if size is None or size < 0:
            size = len(self._payload) - self._position
        chunk = self._payload[self._position : self._position + size]
        self._position += len(chunk)
        return chunk

    def close(self):
        self._on_close()


class CountingBodyFactory(requests.BodyFactory):
    """Body factory that counts stream creations and closes."""

    def __init__(
        self,
        payload,
        content_length=None,
        fail_after_reads=None,
        fail_on_create_call=None,
        stream_class=CountingStream,
    ):
        self.payload = payload
        self.content_length = content_length
        self.fail_after_reads = fail_after_reads
        self.fail_on_create_call = fail_on_create_call
        self.stream_class = stream_class
        self.create_calls = 0
        self.created = 0
        self.closed = 0

    def create(self):
        self.create_calls += 1
        if (
            self.fail_on_create_call is not None
            and self.create_calls == self.fail_on_create_call
        ):
            raise RuntimeError("cannot create another stream")
        self.created += 1
        return self.stream_class(
            self.payload,
            on_close=self._record_close,
            fail_after_reads=self.fail_after_reads,
        )

    def _record_close(self):
        self.closed += 1


class RecordingConnection:
    """Stand-in for a urllib3 connection: reads and records the request
    body of every send, then replies with the next queued response."""

    def __init__(self, responses):
        self.responses = [
            HTTPResponse(
                body=io.BytesIO(b""),
                status=status,
                headers=headers,
                preload_content=False,
                decode_content=False,
            )
            for status, headers in responses
        ]
        # (method, url, headers, body bytes) of each hop, in order.
        self.hops = []

    def urlopen(self, method, url, body=None, headers=None, **kwargs):
        received = b""
        if body is not None:
            while True:
                chunk = body.read(8192)
                if not chunk:
                    break
                received += chunk
        self.hops.append((method, url, dict(headers), received, kwargs))
        return self.responses[len(self.hops) - 1]


def session_sending_to(monkeypatch, connection):
    """A session whose adapter sends every request to ``connection``."""
    session = requests.Session()
    session.trust_env = False
    monkeypatch.setattr(
        HTTPAdapter,
        "get_connection_with_tls_context",
        lambda self, request, verify, proxies=None, cert=None: connection,
    )
    return session


def test_body_factory_redirect_creates_fresh_stream_per_send(monkeypatch):
    """A 307 redirect resends the body: the factory is called once per hop,
    each produced stream is closed exactly once, and every hop receives the
    full payload with the declared Content-Length."""
    payload = b"temporary-stream-bytes" * 4
    factory = CountingBodyFactory(payload, content_length=len(payload))
    connection = RecordingConnection(
        [
            (307, {"Location": "/redirected", "Content-Length": "0"}),
            (200, {"Content-Length": "0"}),
        ]
    )
    session = session_sending_to(monkeypatch, connection)

    r = session.post("http://example.com/upload", data=factory)

    assert r.status_code == 200
    assert factory.create_calls == 2
    assert factory.created == 2
    assert factory.closed == 2

    assert len(connection.hops) == 2
    first_method, first_url, _, first_body, _ = connection.hops[0]
    second_method, second_url, second_headers, second_body, _ = connection.hops[1]
    assert (first_method, first_url) == ("POST", "/upload")
    assert (second_method, second_url) == ("POST", "/redirected")
    assert first_body == payload
    assert second_body == payload
    assert second_headers["Content-Length"] == str(len(payload))


def test_body_factory_without_declared_length_sends_chunked(monkeypatch):
    """A factory that declares no length streams each hop with
    Transfer-Encoding: chunked."""
    payload = b"temporary-stream-bytes" * 4
    factory = CountingBodyFactory(payload)
    connection = RecordingConnection([(200, {"Content-Length": "0"})])
    session = session_sending_to(monkeypatch, connection)

    r = session.post("http://example.com/upload", data=factory)

    assert r.status_code == 200
    assert factory.created == 1
    assert factory.closed == 1

    method, _, headers, body, kwargs = connection.hops[0]
    assert method == "POST"
    assert body == payload
    assert headers["Transfer-Encoding"] == "chunked"
    assert "Content-Length" not in headers
    assert kwargs["chunked"] is True


def test_body_factory_send_failure_closes_created_stream(monkeypatch):
    """If the send fails midway, the stream created for that send is still
    closed exactly once."""
    payload = b"x" * 65536
    factory = CountingBodyFactory(
        payload, content_length=len(payload), fail_after_reads=1
    )
    connection = RecordingConnection([(200, {"Content-Length": "0"})])
    session = session_sending_to(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="stream broke mid-send"):
        session.post("http://example.com/upload", data=factory)

    assert factory.create_calls == 1
    assert factory.created == 1
    assert factory.closed == 1


def test_body_factory_method_change_redirect_drops_body(monkeypatch):
    """A 303 redirect switches to GET: no new stream is created for the
    second hop and the previously created stream is closed exactly once."""
    payload = b"temporary-stream-bytes" * 4
    factory = CountingBodyFactory(payload, content_length=len(payload))
    connection = RecordingConnection(
        [
            (303, {"Location": "/target", "Content-Length": "0"}),
            (200, {"Content-Length": "0"}),
        ]
    )
    session = session_sending_to(monkeypatch, connection)

    r = session.post("http://example.com/upload", data=factory)

    assert r.status_code == 200
    assert factory.create_calls == 1
    assert factory.created == 1
    assert factory.closed == 1

    assert len(connection.hops) == 2
    first_method, _, _, first_body, _ = connection.hops[0]
    second_method, second_url, second_headers, second_body, _ = connection.hops[1]
    assert first_method == "POST"
    assert first_body == payload
    assert (second_method, second_url) == ("GET", "/target")
    assert second_body == b""
    assert "Content-Length" not in second_headers


def test_body_factory_create_failure_on_redirect_closes_prior_streams(monkeypatch):
    """If creating the stream for a redirect resend fails, the stream from
    the earlier hop has already been closed exactly once."""
    payload = b"temporary-stream-bytes" * 4
    factory = CountingBodyFactory(
        payload, content_length=len(payload), fail_on_create_call=2
    )
    connection = RecordingConnection(
        [
            (307, {"Location": "/redirected", "Content-Length": "0"}),
            (200, {"Content-Length": "0"}),
        ]
    )
    session = session_sending_to(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="cannot create another stream"):
        session.post("http://example.com/upload", data=factory)

    assert factory.create_calls == 2
    assert factory.created == 1
    assert factory.closed == 1
    assert len(connection.hops) == 1


def test_body_factory_length_contract_mismatch(monkeypatch):
    """A produced stream that observably violates the declared
    Content-Length is rejected before anything is sent, and is still
    closed exactly once."""

    class LyingStream(CountingStream):
        def __len__(self):
            return 3

    payload = b"temporary-stream-bytes" * 4
    factory = CountingBodyFactory(
        payload, content_length=len(payload), stream_class=LyingStream
    )
    connection = RecordingConnection([(200, {"Content-Length": "0"})])
    session = session_sending_to(monkeypatch, connection)

    with pytest.raises(ValueError, match="Content-Length"):
        session.post("http://example.com/upload", data=factory)

    assert factory.create_calls == 1
    assert factory.created == 1
    assert factory.closed == 1
    assert connection.hops == []
