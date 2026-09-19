"""Harness custom message types ported from ``harness/messages.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Union

from pi_ai.types import (
    ImageContent,
    Message,
    TextContent,
    UserMessage,
)

__all__ = [
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "BashExecutionMessage",
    "CustomMessage",
    "BranchSummaryMessage",
    "CompactionSummaryMessage",
    "bash_execution_to_text",
    "create_branch_summary_message",
    "create_compaction_summary_message",
    "create_custom_message",
    "convert_to_llm",
]

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)

COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"

BRANCH_SUMMARY_SUFFIX = "</summary>"


@dataclass
class BashExecutionMessage:
    role: str = field(default="bashExecution", init=False)
    command: str
    output: str
    exit_code: Optional[int]
    cancelled: bool
    truncated: bool
    full_output_path: Optional[str] = None
    timestamp: int = 0
    exclude_from_context: bool = False


@dataclass
class CustomMessage:
    role: str = field(default="custom", init=False)
    custom_type: str
    content: Any
    display: bool
    details: Any = None
    timestamp: int = 0


@dataclass
class BranchSummaryMessage:
    role: str = field(default="branchSummary", init=False)
    summary: str
    from_id: Optional[str]
    timestamp: int


@dataclass
class CompactionSummaryMessage:
    role: str = field(default="compactionSummary", init=False)
    summary: str
    tokens_before: int
    timestamp: int


def _to_timestamp(timestamp: Union[str, int]) -> int:
    if isinstance(timestamp, int):
        return timestamp
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated and msg.full_output_path:
        text += f"\n\n[Output truncated. Full output: {msg.full_output_path}]"
    return text


def create_branch_summary_message(summary: str, from_id: Optional[str], timestamp: Union[str, int]) -> BranchSummaryMessage:
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=_to_timestamp(timestamp))


def create_compaction_summary_message(summary: str, tokens_before: int, timestamp: Union[str, int]) -> CompactionSummaryMessage:
    return CompactionSummaryMessage(summary=summary, tokens_before=tokens_before, timestamp=_to_timestamp(timestamp))


def create_custom_message(
    custom_type: str,
    content: Any,
    display: bool,
    details: Any,
    timestamp: Union[str, int],
) -> CustomMessage:
    return CustomMessage(
        custom_type=custom_type,
        content=content,
        display=display,
        details=details,
        timestamp=_to_timestamp(timestamp),
    )


def convert_to_llm(messages: List[Any]) -> List[Message]:
    """Map harness AgentMessages onto LLM-compatible messages."""

    def _convert(message: Any) -> Optional[Message]:
        role = getattr(message, "role", None)
        if role == "bashExecution":
            if message.exclude_from_context:
                return None
            return UserMessage(
                content=[TextContent(text=bash_execution_to_text(message))],
                timestamp=message.timestamp,
            )
        if role == "custom":
            content = (
                [TextContent(text=message.content)]
                if isinstance(message.content, str)
                else message.content
            )
            return UserMessage(content=content, timestamp=message.timestamp)
        if role == "branchSummary":
            return UserMessage(
                content=[TextContent(text=BRANCH_SUMMARY_PREFIX + message.summary + BRANCH_SUMMARY_SUFFIX)],
                timestamp=message.timestamp,
            )
        if role == "compactionSummary":
            return UserMessage(
                content=[TextContent(text=COMPACTION_SUMMARY_PREFIX + message.summary + COMPACTION_SUMMARY_SUFFIX)],
                timestamp=message.timestamp,
            )
        if role in ("system", "user", "assistant", "toolResult"):
            return message
        return None

    return [converted for message in messages if (converted := _convert(message)) is not None]
