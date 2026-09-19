"""Tests for context-overflow detection and assistant-message frames.

Ports the behavioural cases of ``packages/ai/test/overflow.test.ts`` and
``packages/ai/test/assistant-message-frame.test.ts``.
"""

from __future__ import annotations

import json

import pytest

from pi_ai.assistant_message_frame import (
    AssistantMessageFrame,
    AssistantMessageFrameEncoder,
    frame_from_json,
    reduce_assistant_message_frames,
)
from pi_ai.overflow import (
    get_overflow_patterns,
    is_context_overflow,
    is_recoverable_length,
)
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def seed() -> AssistantMessage:
    return AssistantMessage(
        api="test-api",
        provider="test-provider",
        model="test-model",
        timestamp=1,
    )


def error_message(text: str, provider: str = "ollama") -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider=provider,
        model="qwen3.5:35b",
        stop_reason="error",
        error_message=text,
        timestamp=1,
    )


def length_stop(
    input_tokens: int,
    cache_read: int,
    output: int,
    cache_write: int = 0,
    provider: str = "test-provider",
    model: str = "test-model",
) -> AssistantMessage:
    message = AssistantMessage(
        api="openai-completions",
        provider=provider,
        model=model,
        stop_reason="length",
        timestamp=1,
    )
    message.usage.input = input_tokens
    message.usage.cache_read = cache_read
    message.usage.cache_write = cache_write
    message.usage.output = output
    message.usage.total_tokens = input_tokens + cache_read + cache_write + output
    return message


def event(event_type: str, **fields: object) -> AssistantMessageEvent:
    return AssistantMessageEvent(type=event_type, **fields)  # type: ignore[arg-type]


def encode_frame(
    encoder: AssistantMessageFrameEncoder, assistant_event: AssistantMessageEvent
) -> AssistantMessageFrame:
    converted = encoder.encode(assistant_event)
    assert converted is not None, f"Expected {assistant_event.type} event to produce a frame"
    return converted


# ---------------------------------------------------------------------------
# Overflow detection
# ---------------------------------------------------------------------------


def test_is_context_overflow_detects_provider_error_messages():
    assert is_context_overflow(
        error_message("400 `prompt too long; exceeded max context length by 100918 tokens`"), 32768
    )
    assert is_context_overflow(
        error_message("prompt is too long: 213462 tokens > 200000 maximum"), 200000
    )
    assert is_context_overflow(
        error_message(
            "Error: 400 Input length (265330) exceeds model's maximum context length (262144)."
        ),
        262144,
    )
    assert is_context_overflow(
        error_message(
            "400 Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens"
        ),
        256000,
    )
    assert not is_context_overflow(error_message("500 `model runner crashed unexpectedly`"), 32768)


def test_is_context_overflow_excludes_throttling_and_rate_limits():
    # Bedrock formats throttling as "Too many tokens", which also matches a generic pattern.
    assert not is_context_overflow(
        error_message("Throttling error: Too many tokens, please wait before trying again."), 200000
    )
    assert not is_context_overflow(
        error_message("Service unavailable: The service is temporarily unavailable."), 200000
    )
    assert not is_context_overflow(error_message("Rate limit exceeded, please retry after 30 seconds."), 200000)
    assert not is_context_overflow(error_message("Too many requests. Please slow down."), 200000)


@pytest.mark.parametrize("provider, expected", [("cerebras", True), ("opencode-go", False)])
def test_is_context_overflow_treats_cerebras_bodyless_errors_per_provider(provider, expected):
    for message in ("400 status code (no body)", "413 status code (no body)"):
        assert is_context_overflow(error_message(message, provider), 131072) is expected


def test_is_context_overflow_detects_silent_overflow_above_the_context_window():
    message = AssistantMessage(api="openai-completions", provider="zai", model="glm", stop_reason="stop")
    message.usage.input = 190000
    message.usage.cache_read = 15000

    assert is_context_overflow(message, 200000)
    assert not is_context_overflow(message, 250000)
    # Silent overflow requires the caller to supply the context window.
    assert not is_context_overflow(message)


def test_is_context_overflow_detects_length_stop_filling_the_context_window():
    # Xiaomi MiMo truncates oversized input, leaving no room for output.
    full = length_stop(input_tokens=58, cache_read=1048512, output=0, provider="xiaomi", model="mimo-v2.5-pro")
    assert is_context_overflow(full, 1048576)
    # Below the 99% fill threshold and with room left to generate.
    assert not is_context_overflow(length_stop(input_tokens=100, cache_read=0, output=0), 200000)
    assert not is_context_overflow(length_stop(input_tokens=1000, cache_read=0, output=4096), 200000)


def test_is_recoverable_length_compares_against_the_desired_output_limit():
    below = length_stop(input_tokens=3, cache_read=253584, output=16)
    assert is_recoverable_length(below, 128000)
    assert not is_recoverable_length(length_stop(input_tokens=4062, cache_read=0, output=1024), 1024)
    assert is_recoverable_length(length_stop(input_tokens=100, cache_read=0, output=0), 128000)
    # A zero desired limit cannot be under-shot, and other stop reasons never qualify.
    assert not is_recoverable_length(length_stop(input_tokens=100, cache_read=0, output=0), 0)
    stopped = seed()
    stopped.usage.output = 0
    assert not is_recoverable_length(stopped, 128000)


def test_get_overflow_patterns_returns_a_fresh_list():
    patterns = get_overflow_patterns()
    assert isinstance(patterns, list)
    assert any(pattern.search("prompt is too long: 213462 tokens > 200000 maximum") for pattern in patterns)
    assert not any(pattern.search("model runner crashed unexpectedly") for pattern in patterns)

    patterns.clear()
    assert len(get_overflow_patterns()) > 0
    assert get_overflow_patterns()[0] is get_overflow_patterns()[0]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_encoder_round_trip_reconstructs_the_assistant_message():
    partial = seed()
    partial.provider_thinking_level = "high"
    encoder = AssistantMessageFrameEncoder()
    frames = [encode_frame(encoder, event("start", partial=partial))]

    partial.content.append(TextContent(text=""))
    frames.append(encode_frame(encoder, event("text_start", content_index=0, partial=partial)))
    for delta in ("Hel", "lo ", "world"):
        block = partial.content[0]
        assert isinstance(block, TextContent)
        block.text += delta
        frames.append(encode_frame(encoder, event("text_delta", content_index=0, delta=delta, partial=partial)))
    text_block = partial.content[0]
    assert isinstance(text_block, TextContent)
    text_block.text_signature = "sig-text"
    frames.append(
        encode_frame(encoder, event("text_end", content_index=0, content="Hello world", partial=partial))
    )

    partial.content.append(ThinkingContent(thinking=""))
    frames.append(encode_frame(encoder, event("thinking_start", content_index=1, partial=partial)))
    thinking_block = partial.content[1]
    assert isinstance(thinking_block, ThinkingContent)
    thinking_block.thinking = "check"
    thinking_block.thinking_signature = "sig-think"
    frames.append(encode_frame(encoder, event("thinking_delta", content_index=1, delta="check", partial=partial)))
    frames.append(encode_frame(encoder, event("thinking_end", content_index=1, content="check", partial=partial)))

    partial.content.append(ToolCall(id="call-1", name="read"))
    frames.append(encode_frame(encoder, event("toolcall_start", content_index=2, partial=partial)))
    partial.content[2].arguments = {"path": "README.md"}  # type: ignore[union-attr]
    frames.append(
        encode_frame(
            encoder, event("toolcall_delta", content_index=2, delta='{"path":"README.md"}', partial=partial)
        )
    )
    frames.append(
        encode_frame(
            encoder,
            event(
                "toolcall_end",
                content_index=2,
                partial=partial,
                tool_call=ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
            ),
        )
    )

    assert [frame.type for frame in frames] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
    ]

    expected = AssistantMessage(
        content=[
            TextContent(text="Hello world", text_signature="sig-text"),
            ThinkingContent(thinking="check", thinking_signature="sig-think"),
            ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
        ],
        api="test-api",
        provider="test-provider",
        model="test-model",
        provider_thinking_level="high",
        timestamp=1,
    )
    assert reduce_assistant_message_frames(frames) == expected

    # The same frames replay from their wire shape, as the durable store yields them.
    wire_frames = [json.loads(json.dumps(frame.to_json())) for frame in frames]
    assert reduce_assistant_message_frames(wire_frames) == expected


def test_text_end_frame_is_authoritative_and_keeps_the_signature():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [encode_frame(encoder, event("start", partial=partial))]

    partial.content.append(TextContent(text="Hello "))
    frames.append(encode_frame(encoder, event("text_start", content_index=0, partial=partial)))
    partial.content[0] = TextContent(text="Hello world", text_signature="sig-text")
    frames.append(encode_frame(encoder, event("text_delta", content_index=0, delta="incorrect", partial=partial)))
    frames.append(
        encode_frame(encoder, event("text_end", content_index=0, content="Hello world", partial=partial))
    )

    assert frames[-1] == AssistantMessageFrame(
        type="text_end", content_index=0, content="Hello world", text_signature="sig-text"
    )
    # The 6 characters covered by the start snapshot are trimmed from the delta.
    assert frames[-2].delta == "ect"
    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content == [TextContent(text="Hello world", text_signature="sig-text")]


def test_encoder_checkpoints_queued_tool_json_without_replaying_covered_deltas():
    # Events are queued first and only then handed to the encoder, so the block
    # start snapshot already carries arguments advanced by later events.
    partial = seed()
    tool_call = ToolCall(id="call", name="write")
    events = [event("start", partial=partial)]
    partial.content.append(tool_call)
    events.append(event("toolcall_start", content_index=0, partial=partial))
    tool_call.arguments = {"path": "README.md"}
    events.append(event("toolcall_delta", content_index=0, delta='{"path":"READ', partial=partial))
    events.append(event("toolcall_delta", content_index=0, delta='ME.md"}', partial=partial))

    encoder = AssistantMessageFrameEncoder()
    frames = []
    for queued in events:
        converted = encoder.encode(queued)
        if converted is not None:
            frames.append(converted)

    assert [frame.type for frame in frames] == ["start", "toolcall_start", "toolcall_checkpoint"]
    assert frames[-1] == AssistantMessageFrame(
        type="toolcall_checkpoint", content_index=0, json='{"path":"README.md"}'
    )
    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content == [ToolCall(id="call", name="write", arguments={"path": "README.md"})]


def test_encoder_resumes_legacy_grammar_tool_json_from_the_initial_arguments():
    partial = seed()
    tool_call = ToolCall(id="call", name="bash", arguments={"input": "a"})
    encoder = AssistantMessageFrameEncoder()
    frames = [encode_frame(encoder, event("start", partial=partial))]
    partial.content.append(tool_call)
    frames.append(encode_frame(encoder, event("toolcall_start", content_index=0, partial=partial)))
    tool_call.arguments = {"input": "ab"}
    frames.append(
        encode_frame(encoder, event("toolcall_delta", content_index=0, delta='{"input":"ab', partial=partial))
    )
    tool_call.arguments = {"input": "abc"}
    frames.append(encode_frame(encoder, event("toolcall_delta", content_index=0, delta='c"}', partial=partial)))

    assert [frame.type for frame in frames[2:]] == ["toolcall_checkpoint", "toolcall_delta"]
    assert frames[2].json == '{"input":"ab'
    assert frames[3].delta == 'c"}'
    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content == [ToolCall(id="call", name="bash", arguments={"input": "abc"})]


def test_encoder_returns_none_for_events_that_carry_no_frame():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    encode_frame(encoder, event("start", partial=partial))
    assert encoder.encode(event("done", reason="stop", message=partial)) is None

    # A pre-generation error is accepted and settled without a frame.
    assert (
        AssistantMessageFrameEncoder().encode(event("error", reason="error", error=error_message("setup failed")))
        is None
    )

    # A delta fully covered by the block-start snapshot produces no frame.
    covering = seed()
    covering.content.append(TextContent(text="Hel"))
    covered_encoder = AssistantMessageFrameEncoder()
    encode_frame(covered_encoder, event("start", partial=covering))
    encode_frame(covered_encoder, event("text_start", content_index=0, partial=covering))
    assert covered_encoder.encode(event("text_delta", content_index=0, delta="He", partial=covering)) is None

    # An empty delta on a caught-up tool call produces no frame.
    other = seed()
    other.content.append(ToolCall(id="call", name="bash"))
    tool_encoder = AssistantMessageFrameEncoder()
    encode_frame(tool_encoder, event("start", partial=other))
    encode_frame(tool_encoder, event("toolcall_start", content_index=0, partial=other))
    assert tool_encoder.encode(event("toolcall_delta", content_index=0, delta="", partial=other)) is None


def test_encoder_rejects_out_of_order_and_mismatched_events():
    with pytest.raises(ValueError, match="done event appears before start"):
        AssistantMessageFrameEncoder().encode(event("done", reason="stop", message=seed()))

    with pytest.raises(ValueError, match="text_delta event appears before start"):
        AssistantMessageFrameEncoder().encode(event("text_delta", content_index=0, delta="x", partial=seed()))

    encoder = AssistantMessageFrameEncoder()
    partial = seed()
    encode_frame(encoder, event("start", partial=partial))
    with pytest.raises(ValueError, match="more than one start event"):
        encoder.encode(event("start", partial=partial))

    terminal_encoder = AssistantMessageFrameEncoder()
    encode_frame(terminal_encoder, event("start", partial=seed()))
    assert terminal_encoder.encode(event("done", reason="stop", message=partial)) is None
    with pytest.raises(ValueError, match="follows a terminal event"):
        terminal_encoder.encode(event("text_delta", content_index=0, delta="x", partial=partial))

    mismatched = seed()
    mismatched.content.append(ThinkingContent(thinking=""))
    kind_encoder = AssistantMessageFrameEncoder()
    encode_frame(kind_encoder, event("start", partial=mismatched))
    with pytest.raises(ValueError, match="text_start event points to thinking block"):
        kind_encoder.encode(event("text_start", content_index=0, partial=mismatched))

    duplicate = seed()
    duplicate.content.append(TextContent(text=""))
    block_encoder = AssistantMessageFrameEncoder()
    encode_frame(block_encoder, event("start", partial=duplicate))
    encode_frame(block_encoder, event("text_start", content_index=0, partial=duplicate))
    with pytest.raises(ValueError, match="block 0 starts more than once"):
        block_encoder.encode(event("text_start", content_index=0, partial=duplicate))

    with pytest.raises(ValueError, match="Invalid assistant message frame contentIndex"):
        block_encoder.encode(event("text_delta", content_index=-1, delta="x", partial=duplicate))


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def test_reducer_supports_interleaved_streams_by_content_index():
    frames = [
        AssistantMessageFrame(type="start", partial=seed()),
        AssistantMessageFrame(type="text_start", content_index=0, content=TextContent()),
        AssistantMessageFrame(type="toolcall_start", content_index=1, tool_call=ToolCall(id="call", name="lookup")),
        AssistantMessageFrame(type="thinking_start", content_index=2, content=ThinkingContent()),
        AssistantMessageFrame(type="text_delta", content_index=0, delta="answer"),
        AssistantMessageFrame(type="toolcall_delta", content_index=1, delta='{"query":"pi"}'),
        AssistantMessageFrame(type="thinking_delta", content_index=2, delta="check"),
        AssistantMessageFrame(
            type="toolcall_end", content_index=1, id="call", name="lookup", arguments={"query": "pi"}
        ),
        AssistantMessageFrame(type="text_end", content_index=0, content="answer"),
        AssistantMessageFrame(type="thinking_end", content_index=2, content="check"),
    ]

    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content == [
        TextContent(text="answer"),
        ToolCall(id="call", name="lookup", arguments={"query": "pi"}),
        ThinkingContent(thinking="check"),
    ]


def test_reducer_treats_end_signature_metadata_including_absence_as_authoritative():
    frames = [
        AssistantMessageFrame(type="start", partial=seed()),
        AssistantMessageFrame(
            type="text_start", content_index=0, content=TextContent(text="", text_signature="stale-text")
        ),
        AssistantMessageFrame(type="text_end", content_index=0, content=""),
        AssistantMessageFrame(
            type="thinking_start",
            content_index=1,
            content=ThinkingContent(thinking="", thinking_signature="stale-thinking", redacted=True),
        ),
        AssistantMessageFrame(type="thinking_end", content_index=1, content="", thinking_signature="", redacted=False),
        AssistantMessageFrame(
            type="toolcall_start",
            content_index=2,
            tool_call=ToolCall(
                id="call", name="read", thought_signature="stale-tool", namespace="stale-namespace"
            ),
        ),
        AssistantMessageFrame(type="toolcall_end", content_index=2, id="call", name="read", arguments={}),
    ]

    reduced = reduce_assistant_message_frames(frames)
    assert reduced is not None
    assert reduced.content == [
        TextContent(text=""),
        ThinkingContent(thinking="", thinking_signature="", redacted=False),
        ToolCall(id="call", name="read"),
    ]


def test_reducer_returns_none_without_a_start_frame():
    assert reduce_assistant_message_frames([]) is None
    assert reduce_assistant_message_frames([AssistantMessageFrame(type="text_delta", content_index=0, delta="x")]) is None


def test_reducer_rejects_invalid_frame_sequences():
    with pytest.raises(ValueError, match="before the start frame"):
        reduce_assistant_message_frames(
            [
                AssistantMessageFrame(type="text_delta", content_index=0, delta="x"),
                AssistantMessageFrame(type="start", partial=seed()),
            ]
        )

    with pytest.raises(ValueError, match="expected text block at index 0, found toolCall"):
        reduce_assistant_message_frames(
            [
                AssistantMessageFrame(type="start", partial=seed()),
                AssistantMessageFrame(
                    type="toolcall_start", content_index=0, tool_call=ToolCall(id="call", name="run")
                ),
                AssistantMessageFrame(type="text_delta", content_index=0, delta="wrong"),
            ]
        )

    with pytest.raises(ValueError, match="follows the end of block"):
        reduce_assistant_message_frames(
            [
                AssistantMessageFrame(type="start", partial=seed()),
                AssistantMessageFrame(type="text_start", content_index=0, content=TextContent()),
                AssistantMessageFrame(type="text_end", content_index=0, content=""),
                AssistantMessageFrame(type="text_end", content_index=0, content=""),
            ]
        )

    with pytest.raises(ValueError, match="would leave a gap"):
        reduce_assistant_message_frames(
            [
                AssistantMessageFrame(type="start", partial=seed()),
                AssistantMessageFrame(type="text_start", content_index=1, content=TextContent()),
            ]
        )


def test_reducer_snapshots_frame_data_and_stays_pure():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    start = encode_frame(encoder, event("start", partial=partial))
    partial.usage.cost.total = 99.0

    nested = ToolCall(id="call", name="run", arguments={"nested": {"value": "original"}})
    partial.content.append(nested)
    tool_start = encode_frame(encoder, event("toolcall_start", content_index=0, partial=partial))
    nested.arguments["nested"]["value"] = "mutated"

    reduced = reduce_assistant_message_frames([start, tool_start])
    assert reduced is not None
    assert reduced.usage.cost.total == 0.0
    assert reduced.content[0].arguments == {"nested": {"value": "original"}}

    reduced.content[0].arguments["nested"] = "changed"
    assert tool_start.tool_call is not None
    assert tool_start.tool_call.arguments == {"nested": {"value": "original"}}


def test_frame_wire_shape_exposes_only_declared_public_fields():
    start = AssistantMessageFrame(type="start", partial=seed())
    start_json = start.to_json()
    assert set(start_json) == {"type", "partial"}
    assert start_json["partial"]["content"] == []
    assert start_json["partial"]["stopReason"] == "pending"

    text_start = AssistantMessageFrame(
        type="text_start", content_index=0, content=TextContent(text="visible", text_signature="text-sig")
    )
    assert text_start.to_json() == {
        "type": "text_start",
        "contentIndex": 0,
        "content": {"type": "text", "text": "visible", "textSignature": "text-sig"},
    }

    tool_start = AssistantMessageFrame(
        type="toolcall_start",
        content_index=2,
        tool_call=ToolCall(
            id="call", name="run", arguments={"value": 1}, thought_signature="tool-sig", namespace="tools"
        ),
    )
    assert tool_start.to_json() == {
        "type": "toolcall_start",
        "contentIndex": 2,
        "toolCall": {
            "type": "toolCall",
            "id": "call",
            "name": "run",
            "arguments": {"value": 1},
            "thoughtSignature": "tool-sig",
            "namespace": "tools",
        },
    }

    checkpoint = AssistantMessageFrame(type="toolcall_checkpoint", content_index=0, json='{"path":"a')
    assert checkpoint.to_json() == {"type": "toolcall_checkpoint", "contentIndex": 0, "json": '{"path":"a'}


def test_frame_wire_mapping_round_trips_every_field():
    start = AssistantMessageFrame(type="start", partial=seed())
    decoded_start = frame_from_json(start.to_json())
    assert decoded_start == start

    tool_end = AssistantMessageFrame(
        type="toolcall_end",
        content_index=3,
        id="call",
        name="write",
        arguments={"path": "README.md"},
        thought_signature="thought",
        namespace="files",
    )
    assert frame_from_json(json.loads(json.dumps(tool_end.to_json()))) == tool_end

    thinking_end = AssistantMessageFrame(
        type="thinking_end", content_index=1, content="[redacted]", thinking_signature="enc", redacted=True
    )
    decoded = frame_from_json(json.loads(json.dumps(thinking_end.to_json())))
    assert decoded.redacted is True
    assert decoded.thinking_signature == "enc"
    assert decoded.content == "[redacted]"
