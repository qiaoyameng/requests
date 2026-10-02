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
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast, overload

from ._internal_utils import to_native_string
from .compat import basestring, str, urljoin, urlparse
from .cookies import extract_cookies_to_jar
from .utils import parse_dict_header

if TYPE_CHECKING:
    from http.cookiejar import CookieJar
    from typing import Any

    from .models import PreparedRequest, Response

CONTENT_TYPE_FORM_URLENCODED: Final = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART: Final = "multipart/form-data"
_DEFAULT_PORTS: Final = {"http": 80, "https": 443}
_DIGEST_CHALLENGE_CACHE_SIZE: Final = 100


@dataclass
class _DigestChallengeState:
    chal: dict[str, str]
    paths: frozenset[str]
    last_nonce: str = ""
    nonce_count: int = 0


def _digest_origin_and_path(url: str) -> tuple[tuple[str, str, int], str] | None:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if parsed.scheme not in _DEFAULT_PORTS or hostname is None:
        return None

    port = parsed.port or _DEFAULT_PORTS[parsed.scheme]
    origin = (parsed.scheme.lower(), hostname.lower(), port)
    normalized_path = urlparse(urljoin(url, parsed.path or "/")).path or "/"
    return origin, normalized_path


def _path_in_digest_scope(path: str, scope: str) -> bool:
    if scope == "/":
        return True
    if path == scope:
        return True
    return path.startswith(scope if scope.endswith("/") else scope + "/")


def _digest_scope_paths(
    chal: dict[str, str], request_url: str
) -> tuple[tuple[str, str, int], tuple[str, ...]] | None:
    request = _digest_origin_and_path(request_url)
    if request is None:
        return None

    origin, request_path = request
    domain = chal.get("domain")
    if domain is None:
        return origin, ("/",)

    paths: set[str] = set()
    for entry in domain.split():
        if not entry.startswith(("http://", "https://", "/")):
            continue

        resolved_url = urljoin(request_url, entry)
        resolved_origin = _digest_origin_and_path(resolved_url)
        if resolved_origin is None or resolved_origin[0] != origin:
            continue
        paths.add(resolved_origin[1])

    return origin, tuple(sorted(paths))


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
    multi_domain: bool
    _thread_local: threading.local
    last_nonce: str
    nonce_count: int
    chal: dict[str, str]
    challenge_states: OrderedDict[
        tuple[tuple[str, str, int], str, frozenset[str]], _DigestChallengeState
    ]
    pos: int | None
    num_401_calls: int | None

    @overload
    def __init__(
        self, username: str, password: str, multi_domain: bool = False
    ) -> None: ...
    @overload
    def __init__(
        self, username: bytes, password: bytes, multi_domain: bool = False
    ) -> None: ...

    def __init__(
        self,
        username: bytes | str,
        password: bytes | str,
        multi_domain: bool = False,
    ) -> None:
        self.username = username
        self.password = password
        self.multi_domain = multi_domain
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
            self._thread_local.challenge_states = OrderedDict()
            self._thread_local.active_key = None
            self._thread_local.stale_retried = False

    def _digest_challenge_state(
        self, url: str
    ) -> tuple[
        tuple[tuple[str, str, int], str, frozenset[str]], _DigestChallengeState
    ] | None:
        parsed = _digest_origin_and_path(url)
        if parsed is None:
            return None

        origin, path = parsed
        matches: list[
            tuple[int, tuple[tuple[str, str, int], str, frozenset[str]]]
        ] = []
        for key in self._thread_local.challenge_states:
            state_origin, _realm, paths = key
            if state_origin != origin or not any(
                _path_in_digest_scope(path, scope) for scope in paths
            ):
                continue
            specificity = max(
                (
                    len(scope) + (0 if scope.endswith("/") else 1)
                    if _path_in_digest_scope(path, scope)
                    else 0
                )
                for scope in paths
            )
            matches.append((specificity, key))

        if not matches:
            return None

        matches.sort(reverse=True, key=lambda match: match[0])
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            return None

        key = matches[0][1]
        state = self._thread_local.challenge_states[key]
        self._thread_local.challenge_states.move_to_end(key)
        return key, state

    def _store_digest_challenge(
        self, chal: dict[str, str | None], url: str
    ) -> tuple[
        tuple[tuple[str, str, int], str, frozenset[str]], _DigestChallengeState
    ] | None:
        realm = chal.get("realm")
        nonce = chal.get("nonce")
        if not realm or not nonce:
            return None

        scope = _digest_scope_paths(cast(dict[str, str], chal), url)
        if scope is None:
            return None

        origin, paths = scope
        path_set = frozenset(paths)
        key = (origin, realm, path_set)
        previous = self._thread_local.challenge_states.get(key)
        if previous is not None and previous.chal.get("nonce") == nonce:
            state = previous
        else:
            state = _DigestChallengeState(cast(dict[str, str], chal), path_set)

        self._thread_local.challenge_states[key] = state
        self._thread_local.challenge_states.move_to_end(key)
        while (
            len(self._thread_local.challenge_states) > _DIGEST_CHALLENGE_CACHE_SIZE
        ):
            self._thread_local.challenge_states.popitem(last=False)
        return key, state

    def _build_digest_header(
        self,
        method: str,
        url: str,
        chal: dict[str, str],
        last_nonce: str,
        nonce_count: int,
    ) -> tuple[str, str, int] | None:
        """
        :rtype: str
        """

        realm = chal["realm"]
        nonce = chal["nonce"]
        qop = chal.get("qop")
        algorithm = chal.get("algorithm")
        opaque = chal.get("opaque")
        hash_utf8 = None

        if algorithm is None:
            _algorithm = "MD5"
        else:
            _algorithm = algorithm.upper()
        # lambdas assume digest modules are imported at the top level
        if _algorithm == "MD5" or _algorithm == "MD5-SESS":

            def md5_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.md5(x, usedforsecurity=False).hexdigest()

            hash_utf8 = md5_utf8
        elif _algorithm == "SHA":

            def sha_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha1(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha_utf8
        elif _algorithm == "SHA-256":

            def sha256_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha256(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha256_utf8
        elif _algorithm == "SHA-512":

            def sha512_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha512(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha512_utf8

        if hash_utf8 is None:
            return None

        def KD(s: str, d: str) -> str:
            return hash_utf8(f"{s}:{d}")

        # XXX not implemented yet
        entdig = None
        p_parsed = urlparse(url)
        #: path is request-uri defined in RFC 2616 which should not be empty
        path = p_parsed.path or "/"
        if p_parsed.query:
            path += f"?{p_parsed.query}"

        A1 = f"{self.username}:{realm}:{self.password}"
        A2 = f"{method}:{path}"

        HA1 = hash_utf8(A1)
        HA2 = hash_utf8(A2)

        if nonce == last_nonce:
            nonce_count += 1
        else:
            nonce_count = 1
        ncvalue = f"{nonce_count:08x}"
        s = str(nonce_count).encode("utf-8")
        s += nonce.encode("utf-8")
        s += time.ctime().encode("utf-8")
        s += os.urandom(8)

        cnonce = hashlib.sha1(s, usedforsecurity=False).hexdigest()[:16]
        if _algorithm == "MD5-SESS":
            HA1 = hash_utf8(f"{HA1}:{nonce}:{cnonce}")  # type: ignore[reportConstantRedefinition]  # RFC 2617 terminology

        if not qop:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        elif qop == "auth" or "auth" in qop.split(","):
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:auth:{HA2}"
            respdig = KD(HA1, noncebit)
        else:
            # XXX handle auth-int.
            return None

        # XXX should the partial digests be encoded too?
        base = (
            f'username="{self.username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{path}", response="{respdig}"'
        )
        if opaque:
            base += f', opaque="{opaque}"'
        if algorithm:
            base += f', algorithm="{algorithm}"'
        if entdig:
            base += f', digest="{entdig}"'
        if qop:
            base += f', qop="auth", nc={ncvalue}, cnonce="{cnonce}"'

        return f"Digest {base}", nonce, nonce_count

    def build_digest_header(self, method: str, url: str) -> str | None:
        """
        :rtype: str
        """

        result = self._build_digest_header(
            method,
            url,
            self._thread_local.chal,
            self._thread_local.last_nonce,
            self._thread_local.nonce_count,
        )
        if result is None:
            return None

        header, nonce, nonce_count = result
        self._thread_local.last_nonce = nonce
        self._thread_local.nonce_count = nonce_count
        return header

    def _build_scoped_digest_header(
        self,
        method: str,
        url: str,
        state: _DigestChallengeState,
    ) -> str | None:
        result = self._build_digest_header(
            method, url, state.chal, state.last_nonce, state.nonce_count
        )
        if result is None:
            return None

        header, nonce, nonce_count = result
        state.last_nonce = nonce
        state.nonce_count = nonce_count
        return header

    def handle_redirect(self, r: Response, **kwargs: Any) -> None:
        """Reset num_401_calls counter on redirects."""
        if r.is_redirect:
            self._thread_local.num_401_calls = 1
            self._thread_local.stale_retried = False
            if self.multi_domain:
                location = r.headers.get("location")
                if location is not None:
                    target = urljoin(r.url, location)
                    active = self._thread_local.active_key
                    state = (
                        self._thread_local.challenge_states.get(active)
                        if active is not None
                        else None
                    )
                    parsed = _digest_origin_and_path(target)
                    in_scope = False
                    if state is not None and active is not None and parsed is not None:
                        origin, path = parsed
                        in_scope = (
                            active[0] == origin
                            and any(
                                _path_in_digest_scope(path, p) for p in state.paths
                            )
                        )
                    if not in_scope:
                        for response in (r, *r.history):
                            response.request.headers.pop("Authorization", None)
                self._thread_local.active_key = None

    def _resend_with_digest(
        self,
        r: Response,
        method: str,
        url: str,
        state: _DigestChallengeState,
        **kwargs: Any,
    ) -> Response:
        # Consume content and release the original connection
        # to allow our new request to reuse the same one.
        r.content
        r.close()
        prep = r.request.copy()
        cookie_jar = cast("CookieJar", prep._cookies)
        extract_cookies_to_jar(cookie_jar, r.request, r.raw)
        prep.prepare_cookies(cookie_jar)

        _digest_auth = self._build_scoped_digest_header(method, url, state)
        if _digest_auth:
            prep.headers["Authorization"] = _digest_auth
        _r = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep
        return _r

    def _handle_multi_domain_401(
        self, r: Response, s_auth: str, **kwargs: Any
    ) -> Response:
        pat = re.compile(r"digest\s+", flags=re.IGNORECASE)
        parsed_chal = parse_dict_header(pat.sub("", s_auth, count=1))
        challenge = self._store_digest_challenge(parsed_chal, r.url)
        if challenge is None:
            self._thread_local.num_401_calls = 1
            return r

        key, state = challenge
        stale = (state.chal.get("stale") or "").lower() == "true"
        first_challenge = self._thread_local.num_401_calls == 1
        stale_retry = (
            stale
            and self._thread_local.num_401_calls in (2, 3)
            and not self._thread_local.stale_retried
        )
        if not first_challenge and not stale_retry:
            self._thread_local.num_401_calls = 1
            return r

        if stale_retry:
            self._thread_local.stale_retried = True
        self._thread_local.num_401_calls += 1
        self._thread_local.active_key = key
        retry = self._resend_with_digest(
            r,
            cast(str, r.request.method),
            cast(str, r.request.url),
            state,
            **kwargs,
        )
        retry = self.handle_401(retry, **kwargs)
        if retry.is_redirect:
            self.handle_redirect(retry, **kwargs)
        return retry

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

        if self._thread_local.pos is not None:
            # Rewind the file position indicator of the body to where
            # it was to resend the request.
            if (seek := getattr(r.request.body, "seek", None)) is not None:
                seek(self._thread_local.pos)
        s_auth = r.headers.get("www-authenticate", "")

        if "digest" not in s_auth.lower():
            self._thread_local.num_401_calls = 1
            return r

        if self.multi_domain:
            return self._handle_multi_domain_401(r, s_auth, **kwargs)

        if self._thread_local.num_401_calls < 2:
            self._thread_local.num_401_calls += 1
            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            self._thread_local.chal = cast(
                dict[str, str], parse_dict_header(pat.sub("", s_auth, count=1))
            )

            # Consume content and release the original connection
            # to allow our new request to reuse the same one.
            r.content
            r.close()
            prep = r.request.copy()
            cookie_jar = cast("CookieJar", prep._cookies)
            extract_cookies_to_jar(cookie_jar, r.request, r.raw)
            prep.prepare_cookies(cookie_jar)

            _digest_auth = self.build_digest_header(
                cast(str, prep.method), cast(str, prep.url)
            )
            if _digest_auth:
                prep.headers["Authorization"] = _digest_auth
            _r = r.connection.send(prep, **kwargs)
            _r.history.append(r)
            _r.request = prep
            return _r

        self._thread_local.num_401_calls = 1
        return r

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        # Initialize per-thread state, if needed
        self.init_per_thread_state()
        self._thread_local.active_key = None
        self._thread_local.stale_retried = False
        if self.multi_domain:
            r.headers.pop("Authorization", None)
            challenge = self._digest_challenge_state(cast(str, r.url))
            if challenge is not None:
                key, state = challenge
                _digest_auth = self._build_scoped_digest_header(
                    cast(str, r.method), cast(str, r.url), state
                )
            else:
                _digest_auth = None
            if _digest_auth:
                r.headers["Authorization"] = _digest_auth
                self._thread_local.active_key = key
        elif self._thread_local.last_nonce:
            _digest_auth = self.build_digest_header(
                cast(str, r.method), cast(str, r.url)
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
                self.multi_domain == getattr(other, "multi_domain", False),
            ]
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other
