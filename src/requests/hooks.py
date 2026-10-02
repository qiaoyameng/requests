"""
requests.hooks
~~~~~~~~~~~~~~

This module provides the capabilities for the Requests hooks system.

Available hooks:

``response``:
    The response generated from a Request.

``response_body``:
    The body of a response has finished downloading, or will never
    finish because reading it failed or the response was closed first.

    This is a *notification* hook: it fires exactly once per response,
    no matter whether the body is consumed automatically
    (``stream=False``), by iterating with ``stream=True``, or by
    accessing ``Response.content`` directly. Its keyword arguments
    describe the outcome:

    ``hop``:
        Index of the response within its redirect chain (``0`` for the
        first response, ``1`` for the first redirect, and so on), so an
        event can be correlated with ``response.history``.

    ``outcome``:
        One of ``"eof"`` (body read to completion), ``"error"``
        (reading the body raised), or ``"close"`` (the response was
        closed before its body was fully consumed). A response that has
        only delivered its headers so far has not finished downloading
        and produces no event.

    ``error``:
        The exception that interrupted the download when ``outcome`` is
        ``"error"``, otherwise ``None``.

    Return values of ``response_body`` callbacks are ignored: the body
    that was already delivered cannot be replaced. A callback that
    calls ``Response.close()`` itself does not retrigger the event.
    Exceptions raised by callbacks are not swallowed; they propagate to
    the code that triggered the event (e.g. the caller of
    ``iter_content``, ``content`` or ``close``). When a callback raises
    while an ``"error"`` outcome is being reported, the callback's
    exception replaces the original read error (which remains available
    as ``__context__``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import _types as _t
    from .models import Response

#: Outcome reported to ``response_body`` hooks when the body was read to
#: completion (normal EOF).
BODY_OUTCOME_EOF = "eof"

#: Outcome reported when reading the body raised an exception.
BODY_OUTCOME_ERROR = "error"

#: Outcome reported when the response was closed before its body was
#: fully consumed.
BODY_OUTCOME_CLOSE = "close"

HOOKS: list[str] = ["response", "response_body"]


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


def dispatch_body_hook(
    key: str,
    hooks: _t.HooksInputType | None,
    hook_data: Response,
    **kwargs: Any,
) -> Response:
    """Dispatches a notification-style hook on a given piece of data.

    Unlike :func:`dispatch_hook`, return values of the callbacks are
    ignored: notification hooks observe the data but cannot replace it.
    Exceptions raised by callbacks propagate to the caller.
    """
    hooks_dict = hooks or {}
    hook_list: Iterable[_t.HookType] | _t.HookType | None = hooks_dict.get(key)
    if hook_list:
        if isinstance(hook_list, Callable):
            hook_list = [hook_list]
        for hook in hook_list:
            hook(hook_data, **kwargs)
    return hook_data
