import re
import threading

import pytest

import requests
from requests.auth import (
    HTTPDigestAuth,
    _DIGEST_CHALLENGE_CACHE_SIZE,
    _digest_origin_and_path,
    _digest_scope_paths,
    _path_in_digest_scope,
)
from requests.models import PreparedRequest
from tests.testserver.server import Server, consume_socket_content


def test_digest_scope_normalizes_default_ports():
    assert _digest_origin_and_path("http://example.com:80/api")[0] == (
        "http",
        "example.com",
        80,
    )
    assert _digest_origin_and_path("https://EXAMPLE.COM:443/api")[0] == (
        "https",
        "example.com",
        443,
    )
    assert _digest_origin_and_path("http://example.com:8080/api")[0] == (
        "http",
        "example.com",
        8080,
    )
    assert _digest_origin_and_path("http://example.com/api/../apix")[1] == "/apix"
    assert _digest_origin_and_path("http://example.com/api/./users")[1] == (
        "/api/users"
    )


@pytest.mark.parametrize(
    ("path", "scope", "expected"),
    [
        ("/api", "/api", True),
        ("/api/users", "/api", True),
        ("/apix", "/api", False),
        ("/api2", "/api", False),
        ("/other", "/api", False),
        ("/api", "/api/", False),
        ("/api/users", "/api/", True),
        ("/", "/", True),
    ],
)
def test_digest_scope_uses_path_boundaries(path, scope, expected):
    assert _path_in_digest_scope(path, scope) is expected


def test_digest_domain_does_not_authorize_external_origin():
    challenge = {
        "realm": "example",
        "nonce": "n",
        "domain": "http://evil.example/steal http://example.com/api",
    }
    origin, paths = _digest_scope_paths(
        challenge, "http://example.com:80/protected"
    )
    assert origin == ("http", "example.com", 80)
    assert paths == ("/api",)


def test_digest_domain_without_valid_origin_does_not_broaden_scope():
    challenge = {
        "realm": "example",
        "nonce": "n",
        "domain": "http://evil.example/steal",
    }
    origin, paths = _digest_scope_paths(challenge, "http://example.com/protected")
    assert origin == ("http", "example.com", 80)
    assert paths == ()


def test_digest_challenge_cache_keeps_separate_domains_realms_and_counts():
    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    auth.init_per_thread_state()

    for index in range(_DIGEST_CHALLENGE_CACHE_SIZE + 1):
        auth._store_digest_challenge(
            {"realm": f"realm-{index}", "nonce": f"nonce-{index}"},
            f"http://example.com/item-{index}",
        )

    states = auth._thread_local.challenge_states
    assert len(states) == _DIGEST_CHALLENGE_CACHE_SIZE
    assert (
        ("http", "example.com", 80),
        "realm-0",
        frozenset({"/"}),
    ) not in states
    assert (
        ("http", "example.com", 80),
        f"realm-{_DIGEST_CHALLENGE_CACHE_SIZE}",
        frozenset({"/"}),
    ) in states

    first = auth._digest_challenge_state("http://example.com/item-98")
    second = auth._digest_challenge_state("http://example.com/item-99")
    assert first is not None and second is not None
    auth._build_scoped_digest_header("GET", "http://example.com/item-98", first[1])
    auth._build_scoped_digest_header(
        "GET", "http://example.com/item-99", second[1]
    )
    auth._build_scoped_digest_header("GET", "http://example.com/item-98", first[1])
    assert first[1].nonce_count == 2
    assert second[1].nonce_count == 1


def test_digest_preemptive_authorization_requires_a_learned_scope():
    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    auth.init_per_thread_state()
    auth._store_digest_challenge(
        {"realm": "api", "nonce": "api-nonce", "domain": "/api"},
        "http://example.com/api/login",
    )

    assert auth._digest_challenge_state("http://example.com/api/users")
    assert auth._digest_challenge_state("http://example.com/apix") is None
    assert auth._digest_challenge_state("http://other.example/api/users") is None


def _challenge(realm, nonce, domain=None):
    domain_header = f', domain="{domain}"' if domain else ""
    return (
        "HTTP/1.1 401 UNAUTHORIZED\r\n"
        "Content-Length: 0\r\n"
        f'WWW-Authenticate: Digest realm="{realm}", nonce="{nonce}", '
        f"qop=auth{domain_header}\r\n\r\n"
    ).encode()


def _request_record(request_content):
    request_line = request_content.split(b"\r\n", 1)[0].decode()
    auth_match = re.search(
        rb"(?:^|\r\n)Authorization:\s*Digest\s+([^\r\n]+)",
        request_content,
        flags=re.IGNORECASE,
    )
    if auth_match is None:
        return request_line, None, None

    auth = auth_match.group(1).decode()
    realm_match = re.search(r'realm="([^"]+)"', auth)
    nc_match = re.search(r"(?:^|,)\s*nc=([0-9a-fA-F]+)", auth)
    return (
        request_line,
        realm_match.group(1) if realm_match else None,
        nc_match.group(1) if nc_match else None,
    )


def _loopback_handler(realm, nonce, records, domain=None):
    challenge = _challenge(realm, nonce, domain)
    ok = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

    def handler(sock):
        for expected_nc in (None, "00000001", "00000002"):
            request_content = consume_socket_content(sock, timeout=0.5)
            records.append(_request_record(request_content))
            sock.send(challenge if expected_nc is None else ok)
        return request_content

    return handler


def test_digestauth_multidomain_isolates_two_loopback_origins():
    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    first_records = []
    second_records = []
    first_done = threading.Event()
    second_done = threading.Event()
    first_server = Server(
        _loopback_handler("first", "first-nonce", first_records),
        requests_to_handle=1,
        wait_to_close_event=first_done,
    )
    second_server = Server(
        _loopback_handler("second", "second-nonce", second_records),
        requests_to_handle=1,
        wait_to_close_event=second_done,
    )

    with first_server as (first_host, first_port), second_server as (
        second_host,
        second_port,
    ):
        with requests.Session() as session:
            session.auth = auth
            first_url = f"http://{first_host}:{first_port}/"
            second_url = f"http://{second_host}:{second_port}/"

            assert session.get(first_url).status_code == 200
            assert session.get(second_url).status_code == 200
            assert session.get(first_url).status_code == 200
            assert session.get(second_url).status_code == 200

    first_done.set()
    second_done.set()
    assert first_records == [
        ("GET / HTTP/1.1", None, None),
        ("GET / HTTP/1.1", "first", "00000001"),
        ("GET / HTTP/1.1", "first", "00000002"),
    ]
    assert second_records == [
        ("GET / HTTP/1.1", None, None),
        ("GET / HTTP/1.1", "second", "00000001"),
        ("GET / HTTP/1.1", "second", "00000002"),
    ]


def test_digestauth_multidomain_isolates_two_realms_and_path_boundaries():
    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    records = []
    done = threading.Event()
    challenge_api = _challenge("api", "api-nonce", "/api")
    challenge_admin = _challenge("admin", "admin-nonce", "/admin")
    ok = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

    def handler(sock):
        responses = (
            challenge_api,
            ok,
            challenge_admin,
            ok,
            ok,
            ok,
            ok,
        )
        for response in responses:
            request_content = consume_socket_content(sock, timeout=0.5)
            records.append(_request_record(request_content))
            sock.send(response)
        return request_content

    server = Server(handler, requests_to_handle=1, wait_to_close_event=done)

    with server as (host, port):
        with requests.Session() as session:
            session.auth = auth
            base_url = f"http://{host}:{port}"
            assert session.get(f"{base_url}/api/login").status_code == 200
            assert session.get(f"{base_url}/admin/login").status_code == 200
            assert session.get(f"{base_url}/apix").status_code == 200
            assert session.get(f"{base_url}/api/next").status_code == 200
            assert session.get(f"{base_url}/admin/next").status_code == 200

    done.set()
    assert records == [
        ("GET /api/login HTTP/1.1", None, None),
        ("GET /api/login HTTP/1.1", "api", "00000001"),
        ("GET /admin/login HTTP/1.1", None, None),
        ("GET /admin/login HTTP/1.1", "admin", "00000001"),
        ("GET /apix HTTP/1.1", None, None),
        ("GET /api/next HTTP/1.1", "api", "00000002"),
        ("GET /admin/next HTTP/1.1", "admin", "00000002"),
    ]


def test_digestauth_multidomain_redirect_strips_out_of_scope_authorization():
    class FakeRequest:
        def __init__(self, url):
            self.url = url
            self.headers = {"Authorization": "Digest stale"}

    class FakeResponse:
        def __init__(self, url, location, history=()):
            self.url = url
            self.request = FakeRequest(url)
            self.headers = {"location": location}
            self.history = list(history)

        @property
        def is_redirect(self):
            return True

    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    auth.init_per_thread_state()
    key, state = auth._store_digest_challenge(
        {"realm": "api", "nonce": "n", "domain": "/api"},
        "http://example.com/api/login",
    )
    auth._thread_local.active_key = key

    initial = FakeResponse("http://example.com/api/login", "/")
    redirect = FakeResponse(
        "http://example.com/api/login",
        "/logout",
        history=[initial],
    )

    auth.handle_redirect(redirect)

    assert "Authorization" not in redirect.request.headers
    assert "Authorization" not in initial.request.headers
    assert auth._thread_local.active_key is None


def test_digest_preemptive_header_is_set_only_for_matching_prepared_request():
    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    auth.init_per_thread_state()
    auth._store_digest_challenge(
        {"realm": "api", "nonce": "api-nonce", "domain": "/api"},
        "http://example.com/api/login",
    )

    allowed = PreparedRequest()
    allowed.method = "GET"
    allowed.url = "http://example.com/api/users"
    allowed.headers = {}
    allowed.body = None
    auth(allowed)

    blocked = PreparedRequest()
    blocked.method = "GET"
    blocked.url = "http://example.com/apix"
    blocked.headers = {"Authorization": "manual"}
    blocked.body = None
    auth(blocked)

    assert allowed.headers["Authorization"].startswith("Digest ")
    assert 'realm="api"' in allowed.headers["Authorization"]
    assert "Authorization" not in blocked.headers


def test_digest_stale_nonce_update_is_isolated_by_protection_domain():
    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    auth.init_per_thread_state()
    api_key, api_state = auth._store_digest_challenge(
        {"realm": "api", "nonce": "old", "qop": "auth", "domain": "/api"},
        "http://example.com/api/login",
    )
    other_key, other_state = auth._store_digest_challenge(
        {"realm": "admin", "nonce": "admin", "qop": "auth", "domain": "/admin"},
        "http://example.com/admin/login",
    )
    auth._build_scoped_digest_header("GET", "http://example.com/api/a", api_state)
    auth._build_scoped_digest_header("GET", "http://example.com/api/b", api_state)
    auth._build_scoped_digest_header(
        "GET", "http://example.com/admin/a", other_state
    )

    updated_key, updated_state = auth._store_digest_challenge(
        {
            "realm": "api",
            "nonce": "new",
            "qop": "auth",
            "stale": "true",
            "domain": "/api",
        },
        "http://example.com/api/login",
    )
    header = auth._build_scoped_digest_header(
        "GET", "http://example.com/api/c", updated_state
    )

    assert updated_key == api_key
    assert updated_state.last_nonce == "new"
    assert updated_state.nonce_count == 1
    assert 'nonce="new"' in header
    assert "nc=00000001" in header
    assert other_key in auth._thread_local.challenge_states
    assert other_state.nonce_count == 1


def test_digest_multidomain_handles_challenge_without_opening_second_scope():
    auth = HTTPDigestAuth("user", "pass", multi_domain=True)
    auth.init_per_thread_state()
    auth._store_digest_challenge(
        {
            "realm": "api",
            "nonce": "api-nonce",
            "qop": "auth",
            "domain": "/api",
        },
        "http://example.com/api/login",
    )

    allowed = PreparedRequest()
    allowed.method = "GET"
    allowed.url = "http://example.com/api/next"
    allowed.headers = {}
    allowed.body = None
    auth(allowed)

    blocked = PreparedRequest()
    blocked.method = "GET"
    blocked.url = "http://example.com/admin"
    blocked.headers = {}
    blocked.body = None
    auth(blocked)

    api_state = auth._digest_challenge_state("http://example.com/api/next")[1]
    assert "Authorization" in allowed.headers
    assert "nc=00000001" in allowed.headers["Authorization"]
    assert "Authorization" not in blocked.headers
    assert api_state.nonce_count == 1
