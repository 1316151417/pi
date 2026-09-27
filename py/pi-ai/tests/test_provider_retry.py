"""Provider transport retry and timeout wiring.

Covers the policy ported from pi-ai ``utils/provider-retry.ts``: which failures
retry, how the delay is chosen, and that the abort signal interrupts a backoff.
The timeout tests assert the option reaches the HTTP client, since an ignored
``timeoutMs`` is indistinguishable from a working one without inspecting it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pi_ai.abort import AbortController
from pi_ai.providers.openai_completions import _http_timeout
from pi_ai.types import Context, Model, SimpleStreamOptions
from pi_ai.utils.provider_retry import (
    DEFAULT_MAX_RETRY_DELAY_MS,
    ProviderHttpError,
    ProviderRetryAborted,
    is_retryable_provider_error,
    retry_delay_ms,
    retry_provider_request,
)


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (529, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
    ],
)
def test_status_code_retry_policy(status, expected):
    """Only 408/409/429 and 5xx retry; other 4xx never do."""
    error = ProviderHttpError(status=status, headers={}, message="boom")
    assert is_retryable_provider_error(error) is expected


def test_transport_failure_without_a_status_retries():
    """Connection/timeout failures carry no status and are retryable."""
    assert is_retryable_provider_error(ProviderHttpError(status=None, headers={}, message="reset"))
    assert is_retryable_provider_error(
        ProviderHttpError(status=None, headers=None, message="socket hang up")
    )


def test_should_retry_header_overrides_the_status_code():
    """The provider's explicit header wins in both directions."""
    retryable_status = ProviderHttpError(status=503, headers={}, message="unavailable")
    assert is_retryable_provider_error(retryable_status) is True
    assert (
        is_retryable_provider_error(
            ProviderHttpError(status=503, headers={"x-should-retry": "false"}, message="unavailable")
        )
        is False
    )
    assert (
        is_retryable_provider_error(
            ProviderHttpError(status=400, headers={"x-should-retry": "true"}, message="bad request")
        )
        is True
    )


# ---------------------------------------------------------------------------
# Delay selection
# ---------------------------------------------------------------------------


def test_retry_after_ms_wins():
    """A millisecond hint is used verbatim."""
    error = ProviderHttpError(status=429, headers={"retry-after-ms": "1500"}, message="slow down")
    assert retry_delay_ms(error, 0) == 1500


def test_retry_after_seconds_is_converted():
    """A seconds hint is converted to milliseconds."""
    error = ProviderHttpError(status=429, headers={"retry-after": "2"}, message="slow down")
    assert retry_delay_ms(error, 0) == 2000


def test_retry_after_http_date_is_resolved_against_now():
    """A past HTTP-date keeps the signed TS delay; the timer clamps it to zero."""
    error = ProviderHttpError(
        status=429,
        headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"},  # far in the past
        message="slow down",
    )
    assert retry_delay_ms(error, 0) < 0


def test_exponential_backoff_grows_and_is_jittered():
    """Without a hint the delay is min(0.5 * 2^i, 8s) scaled down by up to 25%."""
    error = ProviderHttpError(status=503, headers={}, message="unavailable")
    for index, ceiling in enumerate([500, 1000, 2000, 4000, 8000, 8000]):
        observed = [retry_delay_ms(error, index) for _ in range(200)]
        assert all(ceiling * 0.75 <= value <= ceiling for value in observed), (index, observed[:5])


def test_server_delay_above_the_ceiling_fails_instead_of_blocking():
    """A long server-requested delay raises rather than sleeping it out."""
    error = ProviderHttpError(
        status=429, headers={"retry-after": "600"}, message="come back later"
    )
    with pytest.raises(RuntimeError, match="Server requested 600s retry delay"):
        retry_delay_ms(error, 0, max_retry_delay_ms=1000)

    # The ceiling is configurable, and 0 disables it entirely.
    assert retry_delay_ms(error, 0, max_retry_delay_ms=0) == 600_000
    assert retry_delay_ms(error, 0, max_retry_delay_ms=700_000) == 600_000


def test_default_ceiling_is_one_minute():
    assert DEFAULT_MAX_RETRY_DELAY_MS == 60_000


# ---------------------------------------------------------------------------
# Request loop
# ---------------------------------------------------------------------------


async def test_retries_until_success():
    """Each retry re-invokes the request; the successful attempt is returned."""
    attempts = []

    async def request():
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise ProviderHttpError(status=503, headers={}, message="unavailable")
        return "ok"

    assert await retry_provider_request(request, max_retries=5) == "ok"
    assert attempts == [1, 2, 3]


async def test_non_retryable_failure_propagates_immediately():
    """A 4xx without a hint is not retried."""
    attempts = []

    async def request():
        attempts.append(1)
        raise ProviderHttpError(status=400, headers={}, message="bad request")

    with pytest.raises(ProviderHttpError, match="bad request"):
        await retry_provider_request(request, max_retries=5)
    assert len(attempts) == 1


async def test_exhausted_retries_raise_the_last_error():
    """Running out of retries surfaces the final failure."""
    attempts = []

    async def request():
        attempts.append(1)
        raise ProviderHttpError(status=500, headers={}, message=f"boom {len(attempts)}")

    with pytest.raises(ProviderHttpError, match="boom 3"):
        await retry_provider_request(request, max_retries=2)
    assert len(attempts) == 3


async def test_zero_retries_means_one_attempt():
    """max_retries=0 disables retrying entirely."""
    attempts = []

    async def request():
        attempts.append(1)
        raise ProviderHttpError(status=503, headers={}, message="unavailable")

    with pytest.raises(ProviderHttpError):
        await retry_provider_request(request)
    assert len(attempts) == 1


async def test_abort_interrupts_the_backoff_sleep():
    """An abort during the wait stops retrying instead of sleeping it out."""
    controller = AbortController()
    attempts = []

    async def request():
        attempts.append(1)
        raise ProviderHttpError(status=503, headers={"retry-after-ms": "30000"}, message="later")

    async def abort_soon():
        await asyncio.sleep(0.02)
        controller.abort()

    with pytest.raises(ProviderRetryAborted):
        await asyncio.wait_for(
            asyncio.gather(
                retry_provider_request(request, max_retries=3, signal=controller.signal),
                abort_soon(),
            ),
            2,
        )
    assert len(attempts) == 1


async def test_an_already_aborted_signal_skips_the_retry():
    """A signal that fired before the failure prevents another attempt."""
    controller = AbortController()
    controller.abort()
    attempts = []

    async def request():
        attempts.append(1)
        raise ProviderHttpError(status=503, headers={}, message="unavailable")

    with pytest.raises(ProviderRetryAborted):
        await retry_provider_request(request, max_retries=3, signal=controller.signal)
    assert len(attempts) == 1


# ---------------------------------------------------------------------------
# Timeout wiring
# ---------------------------------------------------------------------------


def test_timeout_option_reaches_the_http_client():
    """`timeoutMs` becomes the per-attempt budget; absent means no client limit."""
    assert _http_timeout(SimpleStreamOptions(timeout_ms=1500)) == 1.5
    assert _http_timeout(SimpleStreamOptions(timeout_ms=250)) == 0.25
    assert _http_timeout(SimpleStreamOptions()) is None
    assert _http_timeout(None) is None
    # A zero/sub-millisecond value clamps up rather than disabling the timeout.
    assert _http_timeout(SimpleStreamOptions(timeout_ms=0)) == 0.001


# ---------------------------------------------------------------------------
# End to end against a local server
# ---------------------------------------------------------------------------


class _FlakyServer:
    """Serves the OpenAI SSE shape, failing the first N attempts."""

    def __init__(self, failures: list[tuple[int, dict]], reply: str = "recovered") -> None:
        self.failures = list(failures)
        self.reply = reply
        self.attempts = 0
        handler = self._handler()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("content-length", 0))
                self.rfile.read(length)
                server.attempts += 1
                if server.failures:
                    status, headers = server.failures.pop(0)
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"transient"}}')
                    return
                chunks = [
                    {"id": "c", "choices": [{"delta": {"content": server.reply}}]},
                    {"id": "c", "choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *_args):
                return

        return Handler

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()


def _model(base_url: str) -> Model:
    return Model(
        id="custom",
        name="custom",
        api="openai-completions",
        provider="openai",
        base_url=base_url,
        input=["text"],
        context_window=8000,
        max_tokens=1000,
    )


async def _collect(model: Model, options: SimpleStreamOptions) -> str:
    """Stream one response and return the accumulated text."""
    from pi_ai.providers.openai_completions import stream_openai
    from pi_ai.transcript import normalize_context

    stream = stream_openai(model, normalize_context(Context(messages=[])), options)
    text = ""
    async for event in stream:
        if event.type == "text_delta":
            text += event.delta
    return text


async def test_stream_retries_a_transient_failure_then_succeeds():
    """A 503 is retried inside the provider and the retry's content is delivered."""
    with _FlakyServer([(503, {"retry-after-ms": "10"})]) as server:
        text = await _collect(
            _model(server.url),
            SimpleStreamOptions(api_key="k", max_retries=2, max_retry_delay_ms=5000),
        )
        assert text == "recovered"
        assert server.attempts == 2


async def test_stream_reports_a_non_retryable_failure_without_retrying():
    """A 401 is surfaced as an error and never retried."""
    with _FlakyServer([(401, {})]) as server:
        from pi_ai.providers.openai_completions import stream_openai
        from pi_ai.transcript import normalize_context

        stream = stream_openai(
            _model(server.url),
            normalize_context(Context(messages=[])),
            SimpleStreamOptions(api_key="k", max_retries=3),
        )
        errors = []
        async for event in stream:
            if event.type == "error":
                errors.append(event.error)
        assert server.attempts == 1
        assert len(errors) == 1
        assert errors[0].stop_reason == "error"
        assert "401" in (errors[0].error_message or "")


async def test_stream_surfaces_an_exhausted_retry_budget():
    """When every attempt fails the caller sees the final error, not a hang."""
    with _FlakyServer([(500, {"retry-after-ms": "10"})] * 3) as server:
        from pi_ai.providers.openai_completions import stream_openai
        from pi_ai.transcript import normalize_context

        stream = stream_openai(
            _model(server.url),
            normalize_context(Context(messages=[])),
            SimpleStreamOptions(api_key="k", max_retries=2, max_retry_delay_ms=5000),
        )
        errors = []
        async for event in stream:
            if event.type == "error":
                errors.append(event.error)
        assert server.attempts == 3
        assert len(errors) == 1
        assert "500" in (errors[0].error_message or "")
