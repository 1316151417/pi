"""Request initiator and vision headers from api/github-copilot-headers.ts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..types import ImageContent, Message, ToolResultMessage, UserMessage


def infer_copilot_initiator(messages: Sequence[Message]) -> Literal["user", "agent"]:
    return "agent" if messages and messages[-1].role != "user" else "user"


def has_copilot_vision_input(messages: Sequence[Message]) -> bool:
    return any(
        isinstance(message, (UserMessage, ToolResultMessage))
        and not isinstance(message.content, str)
        and any(isinstance(block, ImageContent) for block in message.content)
        for message in messages
    )


@dataclass(frozen=True)
class CopilotDynamicHeaderParams:
    messages: Sequence[Message]
    has_images: bool


def build_copilot_dynamic_headers(params: CopilotDynamicHeaderParams) -> dict[str, str]:
    headers = {"X-Initiator": infer_copilot_initiator(params.messages), "Openai-Intent": "conversation-edits"}
    if params.has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers
