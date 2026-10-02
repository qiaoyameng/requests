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
from typing import TYPE_CHECKING, Any, Final, cast, overload

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

_DEFAULT_PORTS: Final = {"http": 80, "https": 443}

#: Bound on the number of learned protection spaces cached per thread when
#: using ``HTTPDigestAuth`` in multi-domain mode.
_MAX_CACHED_PROTECTION_SPACES: Final = 32


def _normalized_origin(url: str) -> tuple[str, str, int | None]:
    """Return a ``(scheme, host, port)`` origin tuple for ``url``.

    Default ports are filled in so that, e.g., ``http://example.com`` and
    ``http://example.com:80`` are considered the same origin.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return scheme, host, port


def _domain_covers_path(domain: str, path: str) -> bool:
    """Return True if ``path`` falls inside the protection-space ``domain``.

    Matching happens on path-segment boundaries so that ``/api`` covers
    ``/api`` and ``/api/users`` but never ``/apix``.
    """
    if domain == "/" or path == domain:
        return True
    prefix = domain if domain.endswith("/") else f"{domain}/"
    return path.startswith(prefix)


def _parse_domain_directive(
    directive: str, origin: tuple[str, str, int | None]
) -> tuple[str, ...]:
    """Parse a challenge ``domain`` directive into same-origin abs_paths.

    Absolute URIs naming a different origin are dropped: a domain directive
    must never authorize sending credentials to an external origin.
    """
    paths = []
    for value in directive.split():
        if value.startswith("/"):
            path = value
        else:
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                continue
            if _normalized_origin(value) != origin:
                continue
            path = parsed.path or "/"
        if path not in paths:
            paths.append(path)
    return tuple(paths)


class _DigestProtectionSpace:
    """Learned digest challenge state for one ``(origin, realm)`` pair."""

    __slots__ = ("chal", "domains", "last_nonce", "nonce_count")

    def __init__(self, chal: dict[str, str], domains: tuple[str, ...]) -> None:
        self.chal = chal
        self.domains = domains
        self.last_nonce = ""
        self.nonce_count = 0


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
    """Attaches HTTP Digest Authentication to the given Request object.

    :param username: The username to authenticate with.
    :param password: The password to authenticate with.
    :param multi_domain: When True, learned challenges are cached per
        protection space, keyed by origin and realm, and restricted to the
        URIs named by the challenge's ``domain`` directive. Authorization
        headers are then only sent preemptively to URIs inside an already
        learned protection space; anything else waits for its own 401
        challenge. When False (the default), a single global challenge
        state is kept, preserving the historical behavior.
    """

    username: bytes | str
    password: bytes | str
    _multi_domain: bool
    _thread_local: threading.local
    last_nonce: str
    nonce_count: int
    chal: dict[str, str]
    pos: int | None
    num_401_calls: int | None

    @overload
    def __init__(
        self, username: str, password: str, multi_domain: bool = ...
    ) -> None: ...
    @overload
    def __init__(
        self, username: bytes, password: bytes, multi_domain: bool = ...
    ) -> None: ...

    def __init__(
        self,
        username: bytes | str,
        password: bytes | str,
        multi_domain: bool = False,
    ) -> None:
        self.username = username
        self.password = password
        self._multi_domain = multi_domain
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
            if self._multi_domain:
                self._thread_local.protection_spaces = OrderedDict()

    def _find_protection_space(self, url: str) -> _DigestProtectionSpace | None:
        """Return the learned protection space covering ``url``, if any.

        Only spaces matching the request's origin and covering its path via
        a challenge ``domain`` directive are eligible, so credentials are
        never preemptively sent outside a learned protection space.
        """
        origin = _normalized_origin(url)
        path = urlparse(url).path or "/"
        spaces = self._thread_local.protection_spaces
        best_key = None
        best_len = -1
        for key, space in spaces.items():
            if key[0] != origin:
                continue
            for domain in space.domains:
                if _domain_covers_path(domain, path) and len(domain) > best_len:
                    best_key = key
                    best_len = len(domain)
        if best_key is None:
            return None
        spaces.move_to_end(best_key)
        return spaces[best_key]

    def _learn_protection_space(
        self, url: str, chal: dict[str, str]
    ) -> _DigestProtectionSpace:
        """Record a challenge for the ``(origin, realm)`` it was issued for."""
        origin = _normalized_origin(url)
        realm = chal.get("realm") or ""
        key = (origin, realm)
        spaces = self._thread_local.protection_spaces
        directive = chal.get("domain")
        domains = (
            _parse_domain_directive(directive, origin) if directive else ()
        ) or ("/",)
        space = spaces.get(key)
        if space is None:
            space = _DigestProtectionSpace(chal, domains)
            spaces[key] = space
            while len(spaces) > _MAX_CACHED_PROTECTION_SPACES:
                spaces.popitem(last=False)
        else:
            space.chal = chal
            space.domains = domains
            if (chal.get("stale") or "").lower() == "true":
                # The server rejected this space's nonce; restart its
                # nonce count without touching any other protection space.
                space.last_nonce = ""
                space.nonce_count = 0
            spaces.move_to_end(key)
        return space

    def _activate_protection_space(self, space: _DigestProtectionSpace) -> None:
        self._thread_local.chal = space.chal
        self._thread_local.last_nonce = space.last_nonce
        self._thread_local.nonce_count = space.nonce_count

    def _sync_protection_space(self, space: _DigestProtectionSpace) -> None:
        space.last_nonce = self._thread_local.last_nonce
        space.nonce_count = self._thread_local.nonce_count

    def build_digest_header(self, method: str, url: str) -> str | None:
        """
        :rtype: str
        """

        realm = self._thread_local.chal["realm"]
        nonce = self._thread_local.chal["nonce"]
        qop = self._thread_local.chal.get("qop")
        algorithm = self._thread_local.chal.get("algorithm")
        opaque = self._thread_local.chal.get("opaque")
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

        if not qop:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        elif qop == "auth" or "auth" in qop.split(","):
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:auth:{HA2}"
            respdig = KD(HA1, noncebit)
        else:
            # XXX handle auth-int.
            return None

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
        if entdig:
            base += f', digest="{entdig}"'
        if qop:
            base += f', qop="auth", nc={ncvalue}, cnonce="{cnonce}"'

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

        if self._thread_local.pos is not None:
            # Rewind the file position indicator of the body to where
            # it was to resend the request.
            if (seek := getattr(r.request.body, "seek", None)) is not None:
                seek(self._thread_local.pos)
        s_auth = r.headers.get("www-authenticate", "")

        if "digest" in s_auth.lower() and self._thread_local.num_401_calls < 2:
            self._thread_local.num_401_calls += 1
            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            parsed_chal = parse_dict_header(pat.sub("", s_auth, count=1))
            space = None
            if self._multi_domain:
                space = self._learn_protection_space(
                    cast(str, r.request.url), parsed_chal
                )
                self._activate_protection_space(space)
            else:
                self._thread_local.chal = parsed_chal

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
            if space is not None:
                self._sync_protection_space(space)
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
        if self._multi_domain:
            # Only preempt challenges for URIs inside a protection space
            # we have already learned; everything else waits for its 401.
            space = self._find_protection_space(cast(str, r.url))
            if space is not None:
                self._activate_protection_space(space)
                _digest_auth = self.build_digest_header(
                    cast(str, r.method), cast(str, r.url)
                )
                self._sync_protection_space(space)
                if _digest_auth:
                    r.headers["Authorization"] = _digest_auth
        elif self._thread_local.last_nonce:
            # If we have a saved nonce, skip the 401
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
            ]
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other
