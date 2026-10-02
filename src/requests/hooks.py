"""
requests.hooks
~~~~~~~~~~~~~~

This module provides the capabilities for the Requests hooks system.

Available hooks:

``response``:
    The response generated from a Request.

``download``:
    The response body finished downloading, or the download was
    interrupted. Hook callbacks receive a :class:`DownloadEvent`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import _types as _t
    from .models import Response

HOOKS: list[str] = ["response", "download"]

#: ``download`` outcome: the body was read to a normal EOF.
DOWNLOAD_EOF = "eof"
#: ``download`` outcome: reading the body raised an exception.
DOWNLOAD_ERROR = "error"
#: ``download`` outcome: the response was closed before the body was
#: fully read.
DOWNLOAD_CLOSED = "closed"

DOWNLOAD_OUTCOMES: tuple[str, str, str] = (
    DOWNLOAD_EOF,
    DOWNLOAD_ERROR,
    DOWNLOAD_CLOSED,
)


class DownloadEvent:
    """Payload delivered to ``download`` hooks.

    Fired exactly once per :class:`Response <requests.Response>` when its
    body download terminates, regardless of how the body was consumed
    (``stream=False`` automatic reads, manual ``iter_content`` iteration,
    or direct ``content`` access).

    .. attribute:: response

        The :class:`Response <requests.Response>` this event belongs to.

    .. attribute:: redirect_hop

        Position of the response in its redirect chain: ``0`` for the
        original request, ``1`` for the first redirect, and so on.

    .. attribute:: outcome

        One of ``"eof"`` (body fully read), ``"error"`` (a read raised an
        exception), or ``"closed"`` (the response was closed before the
        body was consumed). Only ``"eof"`` means the download completed.

    .. attribute:: error

        The exception that interrupted the download when ``outcome`` is
        ``"error"``, otherwise ``None``.
    """

    __slots__ = ("response", "redirect_hop", "outcome", "error")

    def __init__(
        self,
        response: Response,
        redirect_hop: int,
        outcome: str,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.redirect_hop = redirect_hop
        self.outcome = outcome
        self.error = error

    @property
    def completed(self) -> bool:
        """True only when the body was fully downloaded (``outcome == "eof"``)."""
        return self.outcome == DOWNLOAD_EOF

    def __repr__(self) -> str:
        return (
            f"<DownloadEvent [{self.outcome}] "
            f"hop={self.redirect_hop} url={self.response.url!r}>"
        )


def default_hooks() -> dict[str, list[_t.HookType]]:
    return {event: [] for event in HOOKS}


# TODO: response is the only one


def dispatch_hook(
    key: str,
    hooks: _t.HooksInputType | None,
    hook_data: Response,
    **kwargs: Any,
) -> Response:
    """Dispatches a hook dictionary on a given piece of data."""
    hooks_dict = hooks or {}
    hook_list: Iterable[_t.HookType] | _t.HookType | None = hooks_dict.get(key)
    if hook_list:
        if isinstance(hook_list, Callable):
            hook_list = [hook_list]
        for hook in hook_list:
            _hook_data = hook(hook_data, **kwargs)
            if _hook_data is not None:
                hook_data = _hook_data
    return hook_data


def dispatch_download_hook(
    hooks: _t.HooksInputType | None,
    event: DownloadEvent,
) -> None:
    """Dispatches ``download`` hooks for a finished or interrupted body.

    Unlike :func:`dispatch_hook`, return values are ignored: a download
    hook cannot replace the body that was already delivered to the caller.

    Exceptions raised by a hook are not swallowed; they propagate to
    whatever code triggered the download event (e.g. the caller of
    ``iter_content``, ``content``, or ``close``).
    """
    hooks_dict = hooks or {}
    hook_list: Iterable[_t.HookType] | _t.HookType | None = hooks_dict.get("download")
    if hook_list:
        if isinstance(hook_list, Callable):
            hook_list = [hook_list]
        for hook in hook_list:
            hook(event)
