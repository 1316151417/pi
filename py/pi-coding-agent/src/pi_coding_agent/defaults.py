"""Default coding-agent thinking levels from ``core/defaults.ts``."""

from pi_agent_core.types import ThinkingLevel

DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"
THINKING_LEVEL_OPTIONS: tuple[ThinkingLevel, ...] = (
    "off", "minimal", "low", "medium", "high", "xhigh", "max",
)

__all__ = ["DEFAULT_THINKING_LEVEL", "THINKING_LEVEL_OPTIONS"]
