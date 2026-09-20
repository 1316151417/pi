"""Custom OpenAI-compatible endpoints through the CLI model plumbing.

`--base-url` (or `OPENAI_BASE_URL`) re-points the OpenAI provider at any
OpenAI-compatible API — DeepSeek, vLLM, Ollama, LM Studio — and `--model` may
name an id the built-in catalog does not know. These tests drive that path
against a local SSE server speaking the real chat-completions wire format, so
they exercise request building, auth, streaming, and usage accounting without
any network access.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pi_agent_core import Agent, AgentInitialState, AgentOptions
from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_simple_cli.cli import build_models, resolve_model


class _OpenAICompatServer:
    """A minimal OpenAI chat-completions SSE server recording every request."""

    def __init__(self, reply: str = "hello from the custom endpoint") -> None:
        self.requests: list[dict] = []
        self.reply = reply
        handler = self._make_handler()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("authorization"),
                        "model": body.get("model"),
                        "messages": body.get("messages"),
                    }
                )
                chunks = [
                    {"id": "cmpl-1", "choices": [{"delta": {"role": "assistant"}}]},
                    {"id": "cmpl-1", "choices": [{"delta": {"content": server.reply}}]},
                    {
                        "id": "cmpl-1",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                    },
                ]
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *_args):  # silence request logging
                return

        return Handler

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "_OpenAICompatServer":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_custom_base_url_and_unknown_model_id_resolve():
    """--base-url re-points the provider and admits unknown model ids."""
    with _OpenAICompatServer() as server:
        models = build_models(
            "openai",
            base_url=server.url,
            api_key="sk-test-custom",
            model_id="deepseek-chat",
        )
        model = resolve_model(models, "openai", "deepseek-chat")
        assert model.base_url == server.url
        assert model.id == "deepseek-chat"
        # The built-in catalog is still there and still re-pointed.
        assert any(m.id != "deepseek-chat" for m in models.get_models("openai"))
        assert all(m.base_url == server.url for m in models.get_models("openai"))


async def test_custom_endpoint_streams_a_real_agent_turn():
    """A full Agent prompt runs against the custom endpoint end to end."""
    with _OpenAICompatServer(reply="deep via openai wire") as server:
        models = build_models(
            "openai",
            base_url=server.url,
            api_key="sk-test-custom",
            model_id="deepseek-chat",
        )
        model = resolve_model(models, "openai", "deepseek-chat")
        agent = Agent(
            AgentOptions(
                stream_fn=models.stream_simple,
                initial_state=AgentInitialState(
                    system_prompt="You are terse.", model=model
                ),
            )
        )
        deltas: list[str] = []

        def listener(event, _signal):
            if event.type == "message_update":
                inner = event.assistant_message_event
                if inner is not None and inner.type == "text_delta":
                    deltas.append(inner.delta)

        agent.subscribe(listener)
        await agent.prompt("hi")

        assert deltas == ["deep via openai wire"]
        final = agent.state.messages[-1]
        assert final.role == "assistant"
        assert final.content[0].text == "deep via openai wire"
        assert final.usage.input == 11 and final.usage.output == 7
        assert final.stop_reason == "stop"

        request = server.requests[0]
        assert request["path"].endswith("/v1/chat/completions")
        assert request["authorization"] == "Bearer sk-test-custom"
        assert request["model"] == "deepseek-chat"
        roles = [message["role"] for message in request["messages"]]
        assert roles == ["system", "user"]


def test_openai_base_url_environment_variable_is_honored(monkeypatch):
    """OPENAI_BASE_URL works as the env-var spelling of --base-url."""
    with _OpenAICompatServer() as server:
        monkeypatch.setenv("OPENAI_BASE_URL", server.url)
        models = build_models("openai", model_id="qwen3-32b")
        model = resolve_model(models, "openai", "qwen3-32b")
        assert model.base_url == server.url


def test_known_openai_models_resolve_without_a_custom_endpoint():
    """Without --base-url, the built-in catalog resolves as before."""
    models = build_models("openai")
    model = resolve_model(models, "openai", None)
    assert model.provider == "openai"
    assert model.base_url.startswith("https://api.openai.com")


@pytest.mark.parametrize(
    "endpoint_model", ["deepseek-chat", "deepseek-reasoner", "ollama/qwen3:32b"]
)
def test_arbitrary_compatible_model_ids_are_admitted(endpoint_model):
    """Any OpenAI-compatible id can be named; each gets its own Model entry."""
    with _OpenAICompatServer() as server:
        models = build_models(
            "openai", base_url=server.url, api_key="k", model_id=endpoint_model
        )
        assert resolve_model(models, "openai", endpoint_model).id == endpoint_model
