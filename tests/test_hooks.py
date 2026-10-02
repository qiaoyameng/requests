import pytest
from urllib3.exceptions import ProtocolError

import requests
from requests import hooks
from requests.adapters import BaseAdapter
from requests.exceptions import ChunkedEncodingError
from requests.models import Response
from requests.sessions import merge_hooks
from requests.structures import CaseInsensitiveDict


def hook(value):
    return value[1:]


@pytest.mark.parametrize(
    "hooks_list, result",
    (
        (hook, "ata"),
        ([hook, lambda x: None, hook], "ta"),
    ),
)
def test_hooks(hooks_list, result):
    assert hooks.dispatch_hook("response", {"response": hooks_list}, "Data") == result


def test_default_hooks():
    assert hooks.default_hooks() == {"response": [], "download": []}


class FakeRaw:
    """A minimal file-like raw object backing a Response."""

    def __init__(self, chunks=(), exc=None):
        self._chunks = list(chunks)
        self._exc = exc
        self.closed = False

    def read(self, chunk_size):
        if self._chunks:
            return self._chunks.pop(0)
        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc
        return b""

    def close(self):
        self.closed = True


def make_response(raw):
    r = Response()
    r.status_code = 200
    r.url = "http://example.com/"
    r.raw = raw
    return r


def collect_events(response):
    events = []
    response.hooks["download"].append(events.append)
    return events


def test_download_event_on_iter_content_eof():
    r = make_response(FakeRaw([b"ab", b"cd"]))
    events = collect_events(r)

    assert b"".join(r.iter_content(2)) == b"abcd"

    assert len(events) == 1
    event = events[0]
    assert event.response is r
    assert event.redirect_hop == 0
    assert event.outcome == hooks.DOWNLOAD_EOF
    assert event.completed
    assert event.error is None


def test_download_event_on_content_access():
    r = make_response(FakeRaw([b"ab", b"cd"]))
    events = collect_events(r)

    assert r.content == b"abcd"
    assert [e.outcome for e in events] == [hooks.DOWNLOAD_EOF]


def test_download_event_fires_only_once():
    r = make_response(FakeRaw([b"ab"]))
    events = collect_events(r)

    assert r.content == b"ab"
    # replaying from the cache and closing must not notify again
    assert b"".join(r.iter_content(1)) == b"ab"
    r.close()

    assert len(events) == 1


def test_download_event_on_read_error():
    failure = ValueError("connection reset")
    r = make_response(FakeRaw([b"ab"], exc=failure))
    events = collect_events(r)

    with pytest.raises(ValueError, match="connection reset"):
        b"".join(r.iter_content(2))

    assert len(events) == 1
    event = events[0]
    assert event.outcome == hooks.DOWNLOAD_ERROR
    assert not event.completed
    assert event.error is failure

    # an interrupted download is not re-reported by close()
    r.close()
    assert len(events) == 1


def test_download_event_on_close_before_consuming():
    raw = FakeRaw([b"ab"])
    r = make_response(raw)
    events = collect_events(r)

    r.close()
    r.close()

    assert raw.closed
    assert len(events) == 1
    assert events[0].outcome == hooks.DOWNLOAD_CLOSED
    assert not events[0].completed


def test_download_hook_close_does_not_recurse():
    r = make_response(FakeRaw([b"ab"]))
    events = []

    def closing_hook(event):
        events.append(event)
        event.response.close()

    r.hooks["download"].append(closing_hook)

    assert r.content == b"ab"
    assert len(events) == 1


def test_download_hook_cannot_replace_content():
    r = make_response(FakeRaw([b"real"]))
    r.hooks["download"].append(lambda event: b"fake")

    assert r.content == b"real"


def test_download_hook_exception_propagates():
    r = make_response(FakeRaw([b"ab"]))

    def broken(event):
        raise RuntimeError("hook failed")

    r.hooks["download"].append(broken)

    with pytest.raises(RuntimeError, match="hook failed"):
        r.content


def test_dispatch_download_hook_ignores_return_values():
    event = hooks.DownloadEvent(
        response=make_response(FakeRaw()), redirect_hop=2, outcome=hooks.DOWNLOAD_EOF
    )
    result = hooks.dispatch_download_hook(
        {"download": [lambda e: "replacement", lambda e: None]}, event
    )
    assert result is None
    assert event.redirect_hop == 2


def test_merge_hooks_preserves_session_download_hooks():
    def download_hook(event):
        pass

    merged = merge_hooks(
        hooks.default_hooks(), {"response": [], "download": [download_hook]}
    )
    assert merged["download"] == [download_hook]


def test_download_events_across_redirect_chain_with_failing_body():
    """Three redirects followed by a body that fails halfway through.

    Every hop records its own event tied to its own response object, and
    the final hop -- which only delivered headers plus a partial body --
    is reported as an error, never as a completed download.
    """

    class EmptyRaw:
        def stream(self, chunk_size, decode_content=True):
            return iter(())

        def close(self):
            pass

        def release_conn(self):
            pass

    class FailingRaw(EmptyRaw):
        def stream(self, chunk_size, decode_content=True):
            yield b"12345"
            raise ProtocolError("Connection broken: IncompleteRead")

    redirects = {
        "http://example.com/": "http://example.com/hop1",
        "http://example.com/hop1": "http://example.com/hop2",
        "http://example.com/hop2": "http://example.com/final",
    }

    class RedirectChainAdapter(BaseAdapter):
        def send(self, request, **kwargs):
            r = Response()
            r.url = request.url
            r.request = request
            r.headers = CaseInsensitiveDict()
            if request.url in redirects:
                r.status_code = 302
                r.headers["Location"] = redirects[request.url]
                r.raw = EmptyRaw()
            else:
                r.status_code = 200
                r.raw = FailingRaw()
            return r

        def close(self):
            pass

    events = []
    session = requests.Session()
    session.trust_env = False
    session.mount("http://", RedirectChainAdapter())
    session.hooks["download"].append(events.append)

    r = session.get("http://example.com/", stream=True)

    # the redirect hops were consumed while resolving; the final
    # response has only delivered its headers so far and must not
    # be counted as a completed (or even terminated) download
    assert [(e.redirect_hop, e.outcome) for e in events] == [
        (0, hooks.DOWNLOAD_EOF),
        (1, hooks.DOWNLOAD_EOF),
        (2, hooks.DOWNLOAD_EOF),
    ]
    assert all(e.response is not r for e in events)

    with pytest.raises(ChunkedEncodingError):
        for _ in r.iter_content(1):
            pass
    r.close()

    assert [(e.redirect_hop, e.outcome) for e in events] == [
        (0, hooks.DOWNLOAD_EOF),
        (1, hooks.DOWNLOAD_EOF),
        (2, hooks.DOWNLOAD_EOF),
        (3, hooks.DOWNLOAD_ERROR),
    ]
    # each event is tied to the response of its own hop
    assert events[0].response is r.history[0]
    assert events[1].response is r.history[1]
    assert events[2].response is r.history[2]
    assert events[3].response is r
    # the failed hop is not a completed download, and close() after the
    # error did not produce a duplicate event
    assert isinstance(events[3].error, ChunkedEncodingError)
    assert not events[3].completed
    assert [e.completed for e in events] == [True, True, True, False]
