"""Core AgentSession path from ``core/agent-session.ts`` and ``core/sdk.ts``."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from pi_agent_core import Agent, AgentInitialState, AgentOptions
from pi_agent_core.types import AgentEvent, AgentTool, ThinkingLevel
from pi_ai.abort import AbortController, AbortSignal
from pi_ai.models import Models, clamp_thinking_level
from pi_ai.transcript import get_current_system_message
from pi_ai.types import ImageContent, Model, SystemMessage, TextContent, UserMessage

from .bash_executor import BashExecutorOptions, BashResult, execute_bash_with_operations
from .compaction import CompactionResult, compact_session, should_compact_session
from .defaults import DEFAULT_THINKING_LEVEL
from .messages import BashExecutionMessage, CustomMessage, convert_to_llm
from .model_runtime import ModelRuntime
from .model_resolver import find_initial_model, restore_model_from_session
from .paths import resolve_path
from .resources import load_project_context_files, load_skills
from .session_manager import SessionManager
from .skills import Skill
from .system_prompt import (
    BuildSystemPromptOptions, build_system_prompt, build_system_prompt_sections,
    diff_system_prompt_sections,
)
from .tools import (
    BashOperations, ToolDefinition, ToolExecutionContext, ToolsOptions,
    create_coding_tool_definitions, create_local_bash_operations, wrap_tool_definition,
)

type AgentSessionEvent = AgentEvent | dict[str, object]
type AgentSessionEventListener = Callable[[AgentSessionEvent], None]


@dataclass
class PromptOptions:
    images: list[ImageContent] | None = None
    streaming_behavior: str | None = None
    preflight_result: Callable[[bool], None] | None = None


@dataclass
class CreateAgentSessionOptions:
    cwd: str | None = None
    models: Models | ModelRuntime | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None
    session_manager: SessionManager | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] = field(default_factory=list)
    no_tools: bool = False
    tool_options: ToolsOptions | None = None
    custom_tools: list[ToolDefinition] = field(default_factory=list)
    custom_prompt: str | None = None
    append_system_prompt: str | None = None
    skill_paths: list[str] = field(default_factory=list)
    no_skills: bool = False
    no_context_files: bool = False


@dataclass
class CreateAgentSessionResult:
    session: AgentSession
    model_fallback_message: str | None = None


class AgentSession:
    def __init__(
        self, agent: Agent, session_manager: SessionManager, models: Models | ModelRuntime,
        cwd: str, definitions: Sequence[ToolDefinition], active_tool_names: Sequence[str],
        custom_prompt: str | None = None, append_system_prompt: str | None = None,
        skills: Sequence[Skill] = (), context_files: Sequence[dict[str, str]] = (),
    ) -> None:
        self.agent = agent
        self.session_manager = session_manager
        self.models = models
        self.cwd = cwd
        self.custom_prompt = custom_prompt
        self.append_system_prompt = append_system_prompt
        self.skills = list(skills)
        self.context_files = [dict(item) for item in context_files]
        self._tool_definitions = {definition.name: definition for definition in definitions}
        self._tool_registry: dict[str, AgentTool] = {}
        for definition in definitions:
            self._tool_registry[definition.name] = wrap_tool_definition(definition, self._tool_context)
        self._listeners: list[AgentSessionEventListener] = []
        self._running = False
        self._steering_messages: list[str] = []
        self._follow_up_messages: list[str] = []
        self._pending_bash_messages: list[BashExecutionMessage] = []
        self._bash_controllers: set[AbortController] = set()
        self._compacting = False
        self._unsubscribe_agent = agent.subscribe(self._handle_agent_event)
        self.set_active_tools_by_name(list(active_tool_names))

    def _tool_context(self) -> ToolExecutionContext:
        return ToolExecutionContext(
            cwd=self.cwd, model=self.agent.state.model,
            session_id=self.session_manager.get_session_id(),
            session_file=self.session_manager.get_session_file(),
            thinking_level=self.agent.state.thinking_level,
        )

    def _prompt_options(self) -> BuildSystemPromptOptions:
        names = self.get_active_tool_names()
        return BuildSystemPromptOptions(
            cwd=self.cwd, custom_prompt=self.custom_prompt,
            append_system_prompt=self.append_system_prompt,
            skills=self.skills, context_files=self.context_files,
            selected_tools=names,
            tool_snippets={
                name: definition.prompt_snippet for name, definition in self._tool_definitions.items()
                if definition.prompt_snippet
            },
            tool_guidelines={
                name: list(definition.prompt_guidelines)
                for name, definition in self._tool_definitions.items()
            },
        )

    def _system_update(self) -> SystemMessage | None:
        previous = get_current_system_message(self.agent.state.messages)
        sections = diff_system_prompt_sections(
            previous.sections if previous and previous.sections is not None else {},
            build_system_prompt_sections(self._prompt_options()),
        )
        if not sections:
            return None
        return SystemMessage(content="", sections=sections, timestamp=int(time.time() * 1000))

    async def _handle_agent_event(self, event: AgentEvent, _signal: AbortSignal) -> None:
        if event.type == "message_start" and getattr(event, "message", None) is not None:
            message = event.message
            if getattr(message, "role", None) == "user":
                content = getattr(message, "content", "")
                text = content if isinstance(content, str) else "".join(
                    block.text for block in content if isinstance(block, TextContent)
                )
                if text in self._steering_messages:
                    self._steering_messages.remove(text)
                    self._emit_queue_update()
                elif text in self._follow_up_messages:
                    self._follow_up_messages.remove(text)
                    self._emit_queue_update()
        self._emit(event)
        if event.type == "message_end":
            message = event.message
            role = getattr(message, "role", None)
            if isinstance(message, CustomMessage):
                self.session_manager.append_custom_message_entry(
                    message.custom_type, message.content, message.display, message.details,
                )
            elif role in ("system", "user", "assistant", "toolResult"):
                self.session_manager.append_message(message)
        elif event.type == "agent_end":
            self._flush_pending_bash_messages()

    def _emit(self, event: AgentSessionEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def _emit_queue_update(self) -> None:
        self._emit({
            "type": "queue_update", "steering": list(self._steering_messages),
            "followUp": list(self._follow_up_messages),
        })

    def subscribe(self, listener: AgentSessionEventListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def dispose(self) -> None:
        self.abort_bash()
        self.agent.abort()
        self._unsubscribe_agent()
        self._listeners.clear()

    @property
    def model(self) -> Model | None:
        return self.agent.state.model

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self.agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        return self._running

    @property
    def is_idle(self) -> bool:
        return not self._running

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(self._prompt_options())

    @property
    def messages(self) -> list[object]:
        return self.agent.state.messages

    @property
    def session_id(self) -> str:
        return self.session_manager.get_session_id()

    @property
    def session_file(self) -> str | None:
        return self.session_manager.get_session_file()

    def get_active_tool_names(self) -> list[str]:
        return [tool.name for tool in self.agent.state.tools]

    def get_tool_definition(self, name: str) -> ToolDefinition | None:
        return self._tool_definitions.get(name)

    def set_active_tools_by_name(self, names: Sequence[str]) -> None:
        self.agent.state.tools = [self._tool_registry[name] for name in names if name in self._tool_registry]

    def _user_message(self, text: str, images: Sequence[ImageContent] | None = None) -> UserMessage:
        return UserMessage(
            content=[TextContent(text=text), *(images or [])],
            timestamp=int(time.time() * 1000),
        )

    async def prompt(self, text: str, options: PromptOptions | None = None) -> None:
        chosen = options or PromptOptions()
        preflight = chosen.preflight_result
        try:
            if self._running:
                if chosen.streaming_behavior == "steer":
                    self.agent.steer(self._user_message(text, chosen.images))
                    self._steering_messages.append(text)
                    self._emit_queue_update()
                elif chosen.streaming_behavior == "followUp":
                    self.agent.follow_up(self._user_message(text, chosen.images))
                    self._follow_up_messages.append(text)
                    self._emit_queue_update()
                else:
                    raise RuntimeError(
                        "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message."
                    )
                if preflight is not None:
                    preflight(True)
                return
            self._flush_pending_bash_messages()
            model = self.model
            if model is None:
                raise RuntimeError("No model selected")
            if not await self.models.is_configured(model.provider):
                raise RuntimeError(f'No API key found for "{model.provider}"')
            messages: list[object] = []
            system_update = self._system_update()
            if system_update is not None:
                messages.append(system_update)
            messages.append(self._user_message(text, chosen.images))
        except Exception:
            if preflight is not None:
                preflight(False)
            raise
        if preflight is not None:
            preflight(True)
        self._running = True
        try:
            await self.agent.prompt(messages)
            if model.context_window > 0 and should_compact_session(self.agent.state.messages, model):
                try:
                    await self.compact(reason="threshold")
                except Exception:
                    pass
        finally:
            self._running = False
            self._flush_pending_bash_messages()
            self._emit({"type": "agent_settled"})

    async def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.agent.steer(self._user_message(text, images))
        self._steering_messages.append(text)
        self._emit_queue_update()

    async def follow_up(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.agent.follow_up(self._user_message(text, images))
        self._follow_up_messages.append(text)
        self._emit_queue_update()

    def abort(self) -> None:
        self.agent.abort()

    async def wait_for_idle(self) -> None:
        await self.agent.wait_for_idle()

    async def compact(
        self, custom_instructions: str | None = None, *, reason: str = "manual",
    ) -> CompactionResult | None:
        if self._compacting:
            raise RuntimeError("Compaction already in progress")
        model = self.model
        if model is None or not isinstance(self.models, ModelRuntime):
            raise RuntimeError("Compaction requires a configured model runtime")
        self._compacting = True
        self._emit({"type": "compaction_start", "reason": reason})
        try:
            result = await compact_session(
                self.session_manager, self.models, model,
                custom_instructions=custom_instructions,
            )
            if result is not None:
                self.agent.state.messages = self.session_manager.build_session_context().messages
            self._emit({"type": "compaction_end", "reason": reason, "result": result})
            return result
        except Exception as error:
            self._emit({"type": "compaction_end", "reason": reason, "error": str(error)})
            raise
        finally:
            self._compacting = False

    def set_model(self, model: Model) -> None:
        self.agent.state.model = model
        self.session_manager.append_model_change(model.provider, model.id)

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        model = self.model
        resolved = cast(ThinkingLevel, clamp_thinking_level(model, level)) if model else "off"
        self.agent.state.thinking_level = resolved
        self.session_manager.append_thinking_level_change(resolved)
        self._emit({"type": "thinking_level_changed", "level": resolved})

    async def execute_bash(
        self, command: str, on_chunk: Callable[[str], None] | None = None,
        *, exclude_from_context: bool = False, operation_id: str | None = None,
        operations: BashOperations | None = None,
    ) -> BashResult:
        controller = AbortController()
        self._bash_controllers.add(controller)

        def on_data(chunk: str) -> None:
            if on_chunk is not None:
                on_chunk(chunk)
            self._emit({"type": "bash_execution_update", "id": operation_id, "delta": chunk})

        try:
            result = await execute_bash_with_operations(
                command, self.session_manager.get_cwd(),
                operations or create_local_bash_operations(),
                BashExecutorOptions(on_chunk=on_data, signal=controller.signal),
            )
            self.record_bash_result(command, result, exclude_from_context=exclude_from_context)
            return result
        finally:
            self._bash_controllers.discard(controller)

    def record_bash_result(self, command: str, result: BashResult, *, exclude_from_context: bool = False) -> None:
        message = BashExecutionMessage(
            command=command, output=result.output, exit_code=result.exit_code,
            cancelled=result.cancelled, truncated=result.truncated,
            full_output_path=result.full_output_path, timestamp=int(time.time() * 1000),
            exclude_from_context=exclude_from_context,
        )
        if self._running:
            self._pending_bash_messages.append(message)
        else:
            self.agent.state.messages.append(message)
            self.session_manager.append_message(message)

    def _flush_pending_bash_messages(self) -> None:
        for message in self._pending_bash_messages:
            self.agent.state.messages.append(message)
            self.session_manager.append_message(message)
        self._pending_bash_messages.clear()

    def abort_bash(self) -> None:
        for controller in list(self._bash_controllers):
            controller.abort()


async def create_agent_session(options: CreateAgentSessionOptions | None = None) -> CreateAgentSessionResult:
    chosen = options or CreateAgentSessionOptions()
    cwd = resolve_path(chosen.cwd or (chosen.session_manager.get_cwd() if chosen.session_manager else os.getcwd()))
    session_manager = chosen.session_manager or SessionManager.create(cwd)
    models = chosen.models
    if models is None:
        models = await ModelRuntime.create()
    previous = session_manager.build_session_context()
    model = chosen.model
    fallback_message: str | None = None
    if model is None and previous.messages and previous.model is not None:
        if isinstance(models, ModelRuntime):
            model, fallback_message = await restore_model_from_session(
                previous.model["provider"], previous.model["modelId"], None, models,
            )
        else:
            model = models.get_model(previous.model["provider"], previous.model["modelId"])
            if model is None or not await models.is_configured(model.provider):
                model = None
                fallback_message = f"Could not restore model {previous.model['provider']}/{previous.model['modelId']}"
    if model is None:
        if isinstance(models, ModelRuntime):
            model = (await find_initial_model(models)).model
        else:
            available = await models.get_available()
            model = available[0] if available else None
        if fallback_message and model is not None:
            fallback_message += f". Using {model.provider}/{model.id}"
    thinking_level = chosen.thinking_level
    if thinking_level is None and previous.messages:
        thinking_level = cast(ThinkingLevel, previous.thinking_level)
    if thinking_level is None:
        thinking_level = DEFAULT_THINKING_LEVEL
    thinking_level = cast(ThinkingLevel, clamp_thinking_level(model, thinking_level)) if model else "off"
    active_names = (
        chosen.tools if chosen.tools is not None else
        [] if chosen.no_tools else ["read", "bash", "edit", "write"]
    )
    active_names = [name for name in active_names if name not in chosen.exclude_tools]
    definitions = [*create_coding_tool_definitions(cwd, chosen.tool_options), *chosen.custom_tools]
    skills = load_skills(cwd, skill_paths=chosen.skill_paths, include_defaults=not chosen.no_skills).skills
    context_files = [] if chosen.no_context_files else load_project_context_files(cwd)
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(
            system_prompt="", model=model, thinking_level=thinking_level,
            tools=[], messages=list(previous.messages),
        ),
        convert_to_llm=convert_to_llm,
        stream_fn=models.stream_simple,
        session_id=session_manager.get_session_id(),
    ))
    if not previous.messages:
        if model is not None:
            session_manager.append_model_change(model.provider, model.id)
        session_manager.append_thinking_level_change(thinking_level)
    session = AgentSession(
        agent, session_manager, models, cwd, definitions, active_names,
        custom_prompt=chosen.custom_prompt,
        append_system_prompt=chosen.append_system_prompt,
        skills=skills, context_files=context_files,
    )
    return CreateAgentSessionResult(session, fallback_message)


__all__ = [
    "AgentSession", "AgentSessionEvent", "AgentSessionEventListener", "PromptOptions",
    "CreateAgentSessionOptions", "CreateAgentSessionResult", "create_agent_session",
]
