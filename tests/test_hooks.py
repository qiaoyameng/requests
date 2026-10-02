import pytest
from urllib3.exceptions import ProtocolError

import requests
from requests import hooks
from requests.adapters import HTTPAdapter
from requests.models import PreparedRequest, Response
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
    assert hooks.default_hooks() == {"response": [], "response_body": []}


def test_dispatch_body_hook_ignores_return_values():
    """Notification hooks observe the data but cannot replace it."""
    calls = []

    def replacing_hook(value, **kwargs):
        calls.append(value)
        return "replacement"

    result = hooks.dispatch_body_hook(
        "response_body", {"response_body": [replacing_hook]}, "Data", hop=0
    )
    assert result == "Data"
    assert calls == ["Data"]


class FakeRaw:
    """A minimal urllib3-like raw response backed by in-memory chunks."""

    def __init__(self, chunks=(), error=None):
        self._chunks = list(chunks)
        self._error = error
        self.closed = False

    def stream(self, chunk_size, decode_content=True):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    def close(self):
        self.closed = True

    def release_conn(self):
        pass


def make_response(url, status=200, chunks=(), error=None, headers=None):
    r = Response()
    r.status_code = status
    r.url = url
    r.headers = CaseInsensitiveDict(headers or {})
    r.raw = FakeRaw(chunks, error)
    req = PreparedRequest()
    req.prepare(method="GET", url=url)
    r.request = req
    return r


def body_event_recorder(events):
    def record(response, hop=None, outcome=None, error=None, **kwargs):
        events.append(
            {
                "hop": hop,
                "outcome": outcome,
                "error": error,
                "response_id": id(response),
                "url": response.url,
            }
        )

    return record


def stub_adapter_send(monkeypatch, send):
    monkeypatch.setattr(HTTPAdapter, "send", lambda self, request, **kw: send(request))


def no_env_session():
    session = requests.Session()
    session.trust_env = False
    return session


def test_response_body_hook_redirect_chain_with_failing_body(monkeypatch):
    """Three redirect hops followed by a body that fails halfway through.

    Every event must be correlated with its response and redirect hop;
    the final hop, which only delivered its headers before the body
    broke, must not be reported as a completed download.
    """
    events = []
    redirects = {
        "http://example.com/": "http://example.com/hop1",
        "http://example.com/hop1": "http://example.com/hop2",
        "http://example.com/hop2": "http://example.com/final",
    }

    def fake_send(request):
        if request.url in redirects:
            return make_response(
                request.url,
                status=302,
                headers={"Location": redirects[request.url]},
            )
        # Delivers 5 of the promised bytes, then the read fails.
        return make_response(
            request.url,
            chunks=[b"hello"],
            error=ProtocolError("Response ended prematurely"),
        )

    stub_adapter_send(monkeypatch, fake_send)
    session = no_env_session()

    r = session.get(
        "http://example.com/",
        stream=True,
        hooks={"response_body": body_event_recorder(events)},
    )

    assert len(r.history) == 3
    # The final hop has only delivered its headers so far: no body
    # event may exist for it yet.
    assert [e for e in events if e["response_id"] == id(r)] == []

    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        b"".join(r.iter_content(1024))

    # The three redirect hops had their (empty) bodies consumed by the
    # redirect handling and each reported EOF exactly once, in order.
    eof_events = [e for e in events if e["outcome"] == "eof"]
    assert [e["hop"] for e in eof_events] == [0, 1, 2]
    assert [e["response_id"] for e in eof_events] == [
        id(resp) for resp in r.history
    ]

    # The final hop reported exactly one event: an interruption, never
    # a completed download.
    final_events = [e for e in events if e["response_id"] == id(r)]
    assert len(final_events) == 1
    assert final_events[0]["hop"] == 3
    assert final_events[0]["outcome"] == "error"
    assert isinstance(
        final_events[0]["error"], requests.exceptions.ChunkedEncodingError
    )


def test_response_body_hook_stream_false_fires_once_and_cannot_replace(monkeypatch):
    """stream=False auto-reading fires EOF exactly once, before get()
    returns, and the callback cannot replace the delivered body."""
    events = []

    def replacing_hook(response, **kwargs):
        body_event_recorder(events)(response, **kwargs)
        return b"replacement"  # must be ignored

    stub_adapter_send(
        monkeypatch,
        lambda request: make_response(request.url, chunks=[b"hello ", b"world"]),
    )
    session = no_env_session()

    r = session.get("http://example.com/", hooks={"response_body": replacing_hook})

    assert r.content == b"hello world"
    assert [(e["hop"], e["outcome"]) for e in events] == [(0, "eof")]
    # Accessing the content again or closing the response must not fire
    # the event a second time.
    r.content
    r.close()
    assert len(events) == 1


def test_response_body_hook_manual_iteration_fires_once(monkeypatch):
    """stream=True manual iteration reports EOF exactly once."""
    events = []
    stub_adapter_send(
        monkeypatch,
        lambda request: make_response(request.url, chunks=[b"hello ", b"world"]),
    )
    session = no_env_session()

    r = session.get(
        "http://example.com/",
        stream=True,
        hooks={"response_body": body_event_recorder(events)},
    )

    assert events == []
    assert b"".join(r.iter_content(4)) == b"hello world"
    assert [(e["hop"], e["outcome"]) for e in events] == [(0, "eof")]
    r.close()
    assert len(events) == 1


def test_response_body_hook_close_outcome_and_no_recursion(monkeypatch):
    """Closing an unread streaming response reports "close" exactly
    once, even if the callback itself calls close() again."""
    events = []

    def closing_hook(response, **kwargs):
        body_event_recorder(events)(response, **kwargs)
        response.close()  # must not retrigger the event

    stub_adapter_send(
        monkeypatch,
        lambda request: make_response(request.url, chunks=[b"hello world"]),
    )
    session = no_env_session()

    r = session.get(
        "http://example.com/", stream=True, hooks={"response_body": closing_hook}
    )

    r.close()
    assert [(e["hop"], e["outcome"]) for e in events] == [(0, "close")]
    r.close()
    assert len(events) == 1


def test_response_body_hook_exception_propagates(monkeypatch):
    """Exceptions raised by body callbacks are not swallowed; they
    propagate to the code that triggered the event."""

    def boom(response, **kwargs):
        raise RuntimeError("hook failed")

    stub_adapter_send(
        monkeypatch,
        lambda request: make_response(request.url, chunks=[b"hello world"]),
    )
    session = no_env_session()

    with pytest.raises(RuntimeError, match="hook failed"):
        session.get("http://example.com/", hooks={"response_body": boom})
