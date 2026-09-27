"""Coding-agent messages and LLM conversion from ``core/messages.ts``."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

from pi_ai.types import (
    AssistantMessage, ImageContent, Message, SystemMessage, TextContent,
    ToolResultMessage, UserMessage,
)

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n"
    "<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n"
    "<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"


@dataclass
class BashExecutionMessage:
    command: str
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    timestamp: int
    full_output_path: str | None = None
    exclude_from_context: bool = False
    role: Literal["bashExecution"] = field(default="bashExecution", init=False)


@dataclass
class CustomMessage[T]:
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool
    timestamp: int
    details: T | None = None
    role: Literal["custom"] = field(default="custom", init=False)


@dataclass
class BranchSummaryMessage:
    summary: str
    from_id: str | None
    timestamp: int
    role: Literal["branchSummary"] = field(default="branchSummary", init=False)


@dataclass
class CompactionSummaryMessage:
    summary: str
    tokens_before: int
    timestamp: int
    role: Literal["compactionSummary"] = field(default="compactionSummary", init=False)


type CodingAgentMessage = (
    Message | BashExecutionMessage | CustomMessage[object]
    | BranchSummaryMessage | CompactionSummaryMessage
)


def bash_execution_to_text(message: BashExecutionMessage) -> str:
    result = f"Ran `{message.command}`\n"
    if message.output:
        result += f"```\n{message.output}\n```"
    else:
        result += "(no output)"
    if message.cancelled:
        result += "\n\n(command cancelled)"
    elif message.exit_code is not None and message.exit_code != 0:
        result += f"\n\nCommand exited with code {message.exit_code}"
    if message.truncated and message.full_output_path:
        result += f"\n\n[Output truncated. Full output: {message.full_output_path}]"
    return result


def _timestamp_millis(timestamp: str) -> int:
    """Stored session timestamps are ISO strings; Date.getTime uses epoch ms."""
    parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return int(parsed.timestamp() * 1000)


def create_branch_summary_message(
    summary: str, from_id: str, timestamp: str,
) -> BranchSummaryMessage:
    return BranchSummaryMessage(summary, from_id, _timestamp_millis(timestamp))


def create_compaction_summary_message(
    summary: str, tokens_before: int, timestamp: str,
) -> CompactionSummaryMessage:
    return CompactionSummaryMessage(summary, tokens_before, _timestamp_millis(timestamp))


def create_custom_message(
    custom_type: str,
    content: str | list[TextContent | ImageContent],
    display: bool,
    details: object | None,
    timestamp: str,
) -> CustomMessage[object]:
    return CustomMessage(custom_type, content, display, _timestamp_millis(timestamp), details)


def convert_to_llm(messages: list[CodingAgentMessage]) -> list[Message]:
    converted: list[Message] = []
    for message in messages:
        if isinstance(message, BashExecutionMessage):
            if not message.exclude_from_context:
                converted.append(UserMessage(
                    content=[TextContent(text=bash_execution_to_text(message))],
                    timestamp=message.timestamp,
                ))
        elif isinstance(message, CustomMessage):
            content = [TextContent(text=message.content)] if isinstance(message.content, str) else message.content
            converted.append(UserMessage(content=content, timestamp=message.timestamp))
        elif isinstance(message, BranchSummaryMessage):
            converted.append(UserMessage(
                content=[TextContent(text=BRANCH_SUMMARY_PREFIX + message.summary + BRANCH_SUMMARY_SUFFIX)],
                timestamp=message.timestamp,
            ))
        elif isinstance(message, CompactionSummaryMessage):
            converted.append(UserMessage(
                content=[TextContent(text=COMPACTION_SUMMARY_PREFIX + message.summary + COMPACTION_SUMMARY_SUFFIX)],
                timestamp=message.timestamp,
            ))
        elif isinstance(message, (SystemMessage, UserMessage, AssistantMessage, ToolResultMessage)):
            converted.append(message)
    return converted


__all__ = [
    "COMPACTION_SUMMARY_PREFIX", "COMPACTION_SUMMARY_SUFFIX",
    "BRANCH_SUMMARY_PREFIX", "BRANCH_SUMMARY_SUFFIX", "BashExecutionMessage",
    "CustomMessage", "BranchSummaryMessage", "CompactionSummaryMessage",
    "CodingAgentMessage", "bash_execution_to_text", "create_branch_summary_message",
    "create_compaction_summary_message", "create_custom_message", "convert_to_llm",
]
