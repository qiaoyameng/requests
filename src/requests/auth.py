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
from typing import TYPE_CHECKING, Any, Final, cast, overload

from ._internal_utils import to_native_string
from .compat import basestring, str, urljoin, urlparse
from .cookies import extract_cookies_to_jar
from .exceptions import UnrewindableBodyError
from .utils import DEFAULT_PORTS, parse_dict_header

if TYPE_CHECKING:
    from http.cookiejar import CookieJar
    from typing import Any

    from .models import PreparedRequest, Response

CONTENT_TYPE_FORM_URLENCODED: Final = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART: Final = "multipart/form-data"


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
            self._thread_local.stale_retries = 0

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
        hash_utf8 = None
        hash_name = None

        if algorithm is None:
            _algorithm = "MD5"
        else:
            _algorithm = algorithm.upper()
        # lambdas assume digest modules are imported at the top level
        if _algorithm == "MD5" or _algorithm == "MD5-SESS":
            hash_name = "md5"

            def md5_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.md5(x, usedforsecurity=False).hexdigest()

            hash_utf8 = md5_utf8
        elif _algorithm == "SHA":
            hash_name = "sha1"

            def sha_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha1(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha_utf8
        elif _algorithm == "SHA-256":
            hash_name = "sha256"

            def sha256_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha256(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha256_utf8
        elif _algorithm == "SHA-512":
            hash_name = "sha512"

            def sha512_utf8(x: str | bytes) -> str:
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha512(x, usedforsecurity=False).hexdigest()

            hash_utf8 = sha512_utf8

        if hash_utf8 is None:
            return None

        def KD(s: str, d: str) -> str:
            return hash_utf8(f"{s}:{d}")

        def entity_digest(entity_body: Any) -> str:
            if entity_body is None:
                return hash_utf8(b"")
            if isinstance(entity_body, (bytes, bytearray, memoryview)):
                return hash_utf8(bytes(entity_body))
            if isinstance(entity_body, str):
                return hash_utf8(entity_body.encode("utf-8"))

            tell = getattr(entity_body, "tell", None)
            seek = getattr(entity_body, "seek", None)
            read = getattr(entity_body, "read", None)
            if not callable(tell) or not callable(seek) or not callable(read):
                raise UnrewindableBodyError(
                    "Cannot compute digest authentication with an unreplayable "
                    "request body. Provide bytes or a seekable binary stream."
                )

            try:
                body_position = tell()
            except OSError as e:
                raise UnrewindableBodyError(
                    "Unable to determine request body position for digest "
                    "authentication."
                ) from e

            assert hash_name is not None
            digest = hashlib.new(hash_name, usedforsecurity=False)
            try:
                seek(body_position)
                while chunk := read(65536):
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise UnrewindableBodyError(
                            "Digest authentication requires binary request body "
                            "streams."
                        )
                    digest.update(chunk)
            except OSError as e:
                raise UnrewindableBodyError(
                    "Unable to replay request body for digest authentication."
                ) from e
            finally:
                try:
                    seek(body_position)
                except OSError as e:
                    raise UnrewindableBodyError(
                        "Unable to restore request body position after digest "
                        "authentication."
                    ) from e

            return digest.hexdigest()

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

        qop_options = {
            option.strip() for option in qop.split(",") if option.strip()
        } if qop else set()

        entdig = None

        if not qop:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        elif "auth" in qop_options:
            entdig = None
            HA2 = hash_utf8(A2)
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:auth:{HA2}"
            respdig = KD(HA1, noncebit)
            qop = "auth"
        elif "auth-int" in qop_options:
            entdig = entity_digest(body)
            HA2 = hash_utf8(f"{method}:{path}:{entdig}")
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:auth-int:{HA2}"
            respdig = KD(HA1, noncebit)
            qop = "auth-int"
        else:
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
            base += f', qop="{qop}", nc={ncvalue}, cnonce="{cnonce}"'

        return f"Digest {base}"

    def handle_redirect(self, r: Response, **kwargs: Any) -> None:
        """Reset num_401_calls counter on redirects."""
        if r.is_redirect:
            self._thread_local.num_401_calls = 1
            self._thread_local.stale_retries = 0
            if r.request is not None:
                self._deregister_for_cross_origin_redirect(r.request, r)

    def _deregister_for_cross_origin_redirect(
        self, request: PreparedRequest, response: Response
    ) -> None:
        location = response.headers.get("location")
        if location is None:
            return
        target = urljoin(response.url, location)
        if self._should_strip_auth(request.url, target):
            request.deregister_hook("response", self.handle_401)

    @staticmethod
    def _should_strip_auth(old_url: str, new_url: str) -> bool:
        old_parsed = urlparse(old_url)
        new_parsed = urlparse(new_url)
        if old_parsed.hostname != new_parsed.hostname:
            return True
        if (
            old_parsed.scheme == "http"
            and old_parsed.port in (80, None)
            and new_parsed.scheme == "https"
            and new_parsed.port in (443, None)
        ):
            return False

        changed_port = old_parsed.port != new_parsed.port
        changed_scheme = old_parsed.scheme != new_parsed.scheme
        default_port = (DEFAULT_PORTS.get(old_parsed.scheme), None)
        if (
            not changed_scheme
            and old_parsed.port in default_port
            and new_parsed.port in default_port
        ):
            return False

        return changed_port or changed_scheme

    def handle_401(self, r: Response, **kwargs: Any) -> Response:
        """
        Takes the given response and tries digest-auth, if needed.

        :rtype: requests.Response
        """

        initial_request = r.request

        while 400 <= r.status_code < 500:
            if self._thread_local.pos is not None:
                # Rewind the file position indicator of the body to where
                # it was to resend the request.
                if (seek := getattr(r.request.body, "seek", None)) is not None:
                    seek(self._thread_local.pos)

            s_auth = r.headers.get("www-authenticate", "")
            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            is_digest_challenge = "digest" in s_auth.lower()
            if not is_digest_challenge:
                break

            chal = parse_dict_header(pat.sub("", s_auth, count=1))
            stale = chal.get("stale", "").lower() == "true"
            stale_retry_available = stale and not self._thread_local.stale_retries
            if self._thread_local.num_401_calls >= 2 and not stale_retry_available:
                break

            if stale:
                self._thread_local.stale_retries += 1
                self._thread_local.num_401_calls = 2
            else:
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
            previous_response = r
            r = r.connection.send(prep, **kwargs)
            r.history.append(previous_response)
            r.request = prep
            if r.is_redirect:
                self._deregister_for_cross_origin_redirect(initial_request, r)
                self._deregister_for_cross_origin_redirect(prep, r)

        if not 400 <= r.status_code < 500:
            self._thread_local.num_401_calls = 1
        return r

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        # Initialize per-thread state, if needed
        self.init_per_thread_state()
        self._thread_local.num_401_calls = 1
        self._thread_local.stale_retries = 0
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
