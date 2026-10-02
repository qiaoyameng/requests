import threading

import pytest

import requests
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


def _digest_challenge(realm, nonce, domain=None):
    challenge = (
        f'nonce="{nonce}", realm="{realm}", qop=auth'
    )
    if domain is not None:
        challenge += f', domain="{domain}"'
    return (
        b"HTTP/1.1 401 UNAUTHORIZED\r\n"
        b"Content-Length: 0\r\n"
        b"Connection: close\r\n"
        b"WWW-Authenticate: Digest " + challenge.encode("utf-8") + b"\r\n\r\n"
    )


_TEXT_200_CONNECTION_CLOSE = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Length: 0\r\n"
    b"Connection: close\r\n\r\n"
)


def _authorization_line(request_content):
    for line in request_content.split(b"\r\n"):
        if line.lower().startswith(b"authorization:"):
            return line
    return None


def test_digestauth_multi_domain_isolates_origins():
    """In multi-domain mode, learned challenges are scoped to the origin and
    realm that issued them: a second origin must wait for its own challenge,
    and a ``domain`` directive naming an external origin must not authorize
    sending credentials there.
    """
    nonce_a = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    nonce_b = b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    def make_handler(realm, nonce, seen, challenge_holder):
        def handler(sock):
            request_content = consume_socket_content(sock, timeout=0.5)
            seen.append(request_content)
            if b"Authorization:" in request_content:
                sock.send(_TEXT_200_CONNECTION_CLOSE)
            else:
                sock.send(challenge_holder["challenge"])
            return request_content

        return handler

    seen_a, seen_b = [], []
    challenge_a, challenge_b = {}, {}
    auth = requests.auth.HTTPDigestAuth("user", "pass", multi_domain=True)

    close_b = threading.Event()
    server_b = Server(
        make_handler(b"realm-b", nonce_b, seen_b, challenge_b),
        requests_to_handle=3,
        wait_to_close_event=close_b,
    )
    with server_b as (host_b, port_b):
        challenge_b["challenge"] = _digest_challenge("realm-b", nonce_b.decode())
        # The domain directive for origin A maliciously names origin B; it
        # must be ignored so credentials are never sent to B because of it.
        challenge_a["challenge"] = _digest_challenge(
            "realm-a",
            nonce_a.decode(),
            domain=f"http://{host_b}:{port_b}/ /a",
        )
        close_a = threading.Event()
        server_a = Server(
            make_handler(b"realm-a", nonce_a, seen_a, challenge_a),
            requests_to_handle=3,
            wait_to_close_event=close_a,
        )
        with server_a as (host_a, port_a):
            url_a = f"http://{host_a}:{port_a}"
            url_b = f"http://{host_b}:{port_b}"
            with requests.Session() as s:
                s.auth = auth
                assert s.get(f"{url_a}/a").status_code == 200
                assert s.get(f"{url_b}/b").status_code == 200
                assert s.get(f"{url_a}/a/more").status_code == 200
                assert s.get(f"{url_b}/b/more").status_code == 200
            close_a.set()
        close_b.set()

    # Origin A: unauthenticated first request, then nc=1 on the retry, then
    # a preemptive request continuing the nonce count.
    assert len(seen_a) == 3
    assert _authorization_line(seen_a[0]) is None
    auth_a1 = _authorization_line(seen_a[1])
    assert b'realm="realm-a"' in auth_a1
    assert b'nonce="%s"' % nonce_a in auth_a1
    assert b"nc=00000001" in auth_a1
    auth_a2 = _authorization_line(seen_a[2])
    assert b'nonce="%s"' % nonce_a in auth_a2
    assert b"nc=00000002" in auth_a2

    # Origin B: no state borrowed from origin A -- the first request carries
    # no Authorization header despite A's challenge naming B in ``domain``,
    # and B's nonce count starts from 1 independently.
    assert len(seen_b) == 3
    assert _authorization_line(seen_b[0]) is None
    auth_b1 = _authorization_line(seen_b[1])
    assert b'realm="realm-b"' in auth_b1
    assert b'nonce="%s"' % nonce_b in auth_b1
    assert b"nc=00000001" in auth_b1
    auth_b2 = _authorization_line(seen_b[2])
    assert b'nonce="%s"' % nonce_b in auth_b2
    assert b"nc=00000002" in auth_b2


def test_digestauth_multi_domain_same_origin_realms_and_path_boundary():
    """On a single origin, two realms keep fully independent nonce counts,
    and a domain of ``/alpha`` must not be widened to cover ``/alphax``.
    """
    nonce_alpha = b"aaaa0000aaaa0000aaaa0000aaaa0000"
    nonce_beta = b"bbbb0000bbbb0000bbbb0000bbbb0000"
    seen = []

    def handler(sock):
        request_content = consume_socket_content(sock, timeout=0.5)
        seen.append(request_content)
        if b"Authorization:" in request_content:
            sock.send(_TEXT_200_CONNECTION_CLOSE)
        else:
            path = request_content.split(b" ", 2)[1]
            if path.startswith(b"/beta"):
                sock.send(
                    _digest_challenge("realm-beta", nonce_beta.decode(), domain="/beta")
                )
            else:
                sock.send(
                    _digest_challenge("realm-alpha", nonce_alpha.decode(), domain="/alpha")
                )
        return request_content

    auth = requests.auth.HTTPDigestAuth("user", "pass", multi_domain=True)
    close_server = threading.Event()
    server = Server(handler, requests_to_handle=8, wait_to_close_event=close_server)

    with server as (host, port):
        base = f"http://{host}:{port}"
        with requests.Session() as s:
            s.auth = auth
            assert s.get(f"{base}/alpha/1").status_code == 200
            assert s.get(f"{base}/beta/1").status_code == 200
            assert s.get(f"{base}/alpha/2").status_code == 200
            assert s.get(f"{base}/beta/2").status_code == 200
            assert s.get(f"{base}/alphax").status_code == 200
        close_server.set()

    assert len(seen) == 8
    # /alpha/1: challenged, then retried with nc=1.
    assert _authorization_line(seen[0]) is None
    assert b'nonce="%s"' % nonce_alpha in _authorization_line(seen[1])
    assert b"nc=00000001" in _authorization_line(seen[1])
    # /beta/1: a different realm on the same origin gets no preemptive
    # Authorization and starts its own nonce count.
    assert _authorization_line(seen[2]) is None
    assert b'nonce="%s"' % nonce_beta in _authorization_line(seen[3])
    assert b"nc=00000001" in _authorization_line(seen[3])
    # /alpha/2 and /beta/2: preemptive, each continuing its own count.
    assert b'nonce="%s"' % nonce_alpha in _authorization_line(seen[4])
    assert b"nc=00000002" in _authorization_line(seen[4])
    assert b'nonce="%s"' % nonce_beta in _authorization_line(seen[5])
    assert b"nc=00000002" in _authorization_line(seen[5])
    # /alphax: "/alpha" must not match "/alphax", so this request waits for
    # its own challenge; the retry continues realm-alpha's count at nc=3.
    assert _authorization_line(seen[6]) is None
    assert b'nonce="%s"' % nonce_alpha in _authorization_line(seen[7])
    assert b"nc=00000003" in _authorization_line(seen[7])


def test_digestauth_multi_domain_cache_limit_and_stale_isolation():
    """The protection-space cache is bounded, and a stale update resets only
    the nonce count of the protection space it was issued for.
    """
    from requests.auth import _MAX_CACHED_PROTECTION_SPACES

    auth = requests.auth.HTTPDigestAuth("user", "pass", multi_domain=True)
    auth.init_per_thread_state()

    for i in range(_MAX_CACHED_PROTECTION_SPACES + 10):
        auth._learn_protection_space(
            f"http://host{i}.example/", {"realm": "r", "nonce": f"n{i}"}
        )
    spaces = auth._thread_local.protection_spaces
    assert len(spaces) == _MAX_CACHED_PROTECTION_SPACES
    # Oldest entries were evicted first.
    assert (("http", "host0.example", 80), "r") not in spaces
    newest = ("http", f"host{_MAX_CACHED_PROTECTION_SPACES + 9}.example", 80)
    assert (newest, "r") in spaces

    # A stale challenge resets only its own protection space's count.
    space_a = auth._learn_protection_space(
        "http://a.example/", {"realm": "ra", "nonce": "na"}
    )
    space_b = auth._learn_protection_space(
        "http://a.example/", {"realm": "rb", "nonce": "nb"}
    )
    space_a.last_nonce, space_a.nonce_count = "na", 7
    space_b.last_nonce, space_b.nonce_count = "nb", 3
    auth._learn_protection_space(
        "http://a.example/", {"realm": "ra", "nonce": "na2", "stale": "true"}
    )
    assert space_a.chal["nonce"] == "na2"
    assert space_a.last_nonce == ""
    assert space_a.nonce_count == 0
    assert space_b.last_nonce == "nb"
    assert space_b.nonce_count == 3


def test_digestauth_domain_path_boundary_and_port_normalization():
    """Domain matching is segment-aware, default ports are normalized, and
    cross-origin domain entries are dropped.
    """
    from requests.auth import (
        _domain_covers_path,
        _normalized_origin,
        _parse_domain_directive,
    )

    assert _domain_covers_path("/api", "/api")
    assert _domain_covers_path("/api", "/api/users")
    assert _domain_covers_path("/api/", "/api/users")
    assert _domain_covers_path("/", "/anything")
    assert not _domain_covers_path("/api", "/apix")
    assert not _domain_covers_path("/api", "/apiX")
    assert not _domain_covers_path("/api", "/ap")

    assert _normalized_origin("http://example.com/") == ("http", "example.com", 80)
    assert _normalized_origin("http://example.com:80/") == ("http", "example.com", 80)
    assert _normalized_origin("https://EXAMPLE.com/path") == (
        "https", "example.com", 443,
    )
    assert _normalized_origin("http://example.com:8080/") == (
        "http", "example.com", 8080,
    )

    origin = ("http", "example.com", 80)
    assert _parse_domain_directive("/a http://example.com/b", origin) == ("/a", "/b")
    assert _parse_domain_directive("http://example.com:80/b", origin) == ("/b",)
    # External origins (including the same host on a non-default port) are
    # never authorized by a domain directive.
    external = "http://evil.example/ http://example.com:8080/x"
    assert _parse_domain_directive(external, origin) == ()


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
