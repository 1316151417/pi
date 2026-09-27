"""Terminal presentation for the coding-agent session."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from pi_ai.types import AssistantMessage, TextContent, UserMessage
from pi_tui import (
    Box, Component, Editor, EditorTheme, Markdown, MarkdownTheme, ProcessTerminal,
    SelectListTheme, Spacer, Text, TuiInputListenerResult, TuiMainScreen, matches_key,
)

from .agent_session import AgentSession


DEFAULT_APP_KEYBINDINGS: dict[str, str] = {
    "codingAgent.cancel": "ctrl+c",
    "codingAgent.quit": "ctrl+d",
}


def _fg(code: int):
    return lambda value: f"\x1b[38;5;{code}m{value}\x1b[39m"


def _bg(code: int):
    return lambda value: f"\x1b[48;5;{code}m{value}\x1b[49m"


_ACCENT = _fg(116)
_DIM = _fg(244)
_MUTED = _fg(249)
_SUCCESS = _fg(150)
_ERROR = _fg(167)
_WARNING = _fg(226)


_SELECT_THEME = SelectListTheme(
    selected_prefix=_ACCENT, selected_text=_ACCENT, description=_DIM,
    scroll_info=_DIM, no_match=_WARNING,
)
_EDITOR_THEME = EditorTheme(border_color=_ACCENT, select_list=_SELECT_THEME)
_MARKDOWN_THEME = MarkdownTheme(
    heading=_fg(222), link=_fg(110), link_url=_DIM, code=_ACCENT,
    code_block=_SUCCESS, code_block_border=_DIM, quote=_MUTED,
    quote_border=_DIM, hr=_DIM, list_bullet=_ACCENT,
    bold=lambda value: f"\x1b[1m{value}\x1b[22m",
    italic=lambda value: f"\x1b[3m{value}\x1b[23m",
    strikethrough=lambda value: f"\x1b[9m{value}\x1b[29m",
    underline=lambda value: f"\x1b[4m{value}\x1b[24m",
)


def _message_text(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.text for block in content if isinstance(block, TextContent))
    return ""


def _tool_preview(result: object) -> str:
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        return ""
    value = "\n".join(block.text for block in content if isinstance(block, TextContent)).strip()
    return value[:800] + ("..." if len(value) > 800 else "")


class InteractiveMode:
    def __init__(self, session: AgentSession, keybindings: Mapping[str, str] | None = None) -> None:
        self.session = session
        self.loop = asyncio.get_running_loop()
        self.closed = asyncio.Event()
        self.keybindings = {**DEFAULT_APP_KEYBINDINGS, **(keybindings or {})}
        self.tui = TuiMainScreen(ProcessTerminal(), keybindings=self.keybindings)
        self.editor = Editor(self.tui, _EDITOR_THEME)
        self.editor.on_submit = self._on_submit
        self.status = Text("", padding_x=1, padding_y=0)
        self._busy = False
        self._prompt_task: asyncio.Task[None] | None = None
        self._assistant: Markdown | None = None
        self._assistant_text = ""
        self._tool_labels: dict[str, Text] = {}
        self._tool_outputs: dict[str, Text] = {}

        self.tui.add_child(Spacer(1))
        self.tui.add_child(Text(f"{_ACCENT('pi-python')}  {_DIM('/help commands · Ctrl-C interrupt · Ctrl-D exit')}", padding_x=1, padding_y=0))
        self.tui.add_child(Spacer(1))
        self._restore_messages()
        self.tui.add_child(self.status)
        self.tui.add_child(self.editor)
        self.tui.set_focus(self.editor)
        self._set_status()

    def _append_component(self, component: Component) -> None:
        self.tui.remove_child(self.status)
        self.tui.remove_child(self.editor)
        self.tui.add_child(component)
        self.tui.add_child(self.status)
        self.tui.add_child(self.editor)
        self.tui.request_render()

    def _append_text(self, value: str) -> Text:
        component = Text(value, padding_x=1, padding_y=0)
        self._append_component(component)
        return component

    def _append_user(self, value: str) -> None:
        box = Box(1, 1, _bg(236))
        box.add_child(Markdown(value, 0, 0, _MARKDOWN_THEME))
        self._append_component(box)

    def _append_markdown(self, value: str) -> Markdown:
        component = Markdown(value, 1, 0, _MARKDOWN_THEME)
        self._append_component(Spacer(1))
        self._append_component(component)
        return component

    def _restore_messages(self) -> None:
        for message in self.session.messages:
            if isinstance(message, UserMessage):
                value = _message_text(message)
                if value:
                    self.editor.add_to_history(value)
                    self._append_user(value)
            elif isinstance(message, AssistantMessage):
                value = _message_text(message)
                if value:
                    self._append_markdown(value)

    def _set_status(self, detail: str | None = None) -> None:
        model = self.session.model
        name = f"{model.provider}/{model.id}" if model else "no model"
        state = detail or ("working · Ctrl-C abort" if self._busy else "ready · Enter send · Ctrl-D exit")
        self.status.set_text(f"{_ACCENT(name)}  {_DIM('·')}  {_MUTED(state)}")
        self.tui.request_render()

    def _on_input(self, data: str) -> TuiInputListenerResult | None:
        if matches_key(data, self.keybindings["codingAgent.cancel"]):
            self.loop.call_soon_threadsafe(self._cancel)
            return TuiInputListenerResult(consume=True)
        if not self.editor.get_text() and matches_key(data, self.keybindings["codingAgent.quit"]):
            self.loop.call_soon_threadsafe(self._quit)
            return TuiInputListenerResult(consume=True)
        return None

    def _cancel(self) -> None:
        if self._busy:
            self.session.abort()
            self._set_status("aborting")
        elif self.editor.get_text():
            self.editor.set_text("")
            self.tui.request_render()
        else:
            self._quit()

    def _quit(self) -> None:
        if self._busy:
            self.session.abort()
        self.closed.set()

    def _on_submit(self, value: str) -> None:
        self.loop.call_soon_threadsafe(self._submit, value)

    def _submit(self, value: str) -> None:
        prompt = value.strip()
        if not prompt:
            return
        self.editor.add_to_history(prompt)
        if prompt in ("/exit", "/quit"):
            self._quit()
            return
        if prompt == "/help":
            self._append_text("/help · /model [provider/id] · /compact · /exit · Ctrl-C abort")
            return
        if prompt.startswith("/model"):
            self._model_command(prompt)
            return
        if prompt == "/compact":
            if self._busy:
                self._append_text("Wait for the current response before compacting.")
            else:
                self._busy = True
                self._set_status("compacting")
                self._prompt_task = asyncio.create_task(self._compact())
            return
        self._append_user(prompt)
        if self._busy:
            self.session.follow_up(prompt)
            self._set_status("follow-up queued")
            return
        self._busy = True
        self._set_status()
        self._prompt_task = asyncio.create_task(self._prompt(prompt))

    def _model_command(self, command: str) -> None:
        if self._busy:
            self._append_text("Wait for the current response before changing models.")
            return
        parts = command.split(maxsplit=1)
        if len(parts) == 1:
            snapshot = getattr(self.session.models, "get_available_snapshot", None)
            available = snapshot() if callable(snapshot) else self.session.models.get_models()
            choices = [f"{model.provider}/{model.id}" for model in available]
            suffix = f" ... ({len(choices)} total)" if len(choices) > 20 else ""
            self._append_text(", ".join(choices[:20]) + suffix if choices else "No models found.")
            return
        reference = parts[1].strip()
        if "/" not in reference:
            self._append_text("Use /model provider/model-id")
            return
        provider, model_id = reference.split("/", 1)
        model = self.session.models.get_model(provider, model_id)
        if model is None:
            self._append_text(f"Model not found: {reference}")
            return
        self.session.set_model(model)
        self._set_status()

    async def _compact(self) -> None:
        try:
            result = await self.session.compact()
            self._append_text("Context compacted." if result is not None else "Nothing to compact.")
        except Exception as error:
            self._append_text(f"Compaction failed: {error}")
        finally:
            self._busy = False
            self._set_status()

    async def _prompt(self, prompt: str) -> None:
        try:
            await self.session.prompt(prompt)
        except Exception as error:
            self._append_text(f"Error: {error}")
        finally:
            self._busy = False
            self._set_status()

    def _on_event(self, event: object) -> None:
        if isinstance(event, Mapping):
            return
        kind = getattr(event, "type", "")
        if kind == "message_start" and isinstance(getattr(event, "message", None), AssistantMessage):
            self._assistant_text = ""
            self._assistant = self._append_markdown("")
        elif kind == "message_update":
            update = getattr(event, "assistant_message_event", None)
            if update is not None and update.type == "thinking_delta":
                self._set_status("thinking · Ctrl-C abort")
            elif update is not None and update.type == "text_delta" and update.delta:
                if self._assistant is None:
                    self._assistant = self._append_markdown("")
                self._assistant_text += update.delta
                self._assistant.set_text(self._assistant_text)
                self.tui.request_render()
        elif kind == "message_end" and isinstance(getattr(event, "message", None), AssistantMessage):
            message = event.message
            final_text = _message_text(message)
            if final_text:
                if self._assistant is None:
                    self._assistant = self._append_markdown(final_text)
                else:
                    self._assistant.set_text(final_text)
            if message.stop_reason in ("error", "aborted"):
                self._append_text(message.error_message or f"Request {message.stop_reason}")
            if message.usage and (message.usage.input or message.usage.output):
                self._append_text(f"tokens in={message.usage.input} out={message.usage.output}")
            self._assistant = None
            self.tui.request_render()
        elif kind == "tool_execution_start":
            name = getattr(event, "tool_name", "tool")
            tool_id = getattr(event, "tool_call_id", "")
            args = json.dumps(getattr(event, "args", None), ensure_ascii=False, default=str)
            summary = args if len(args) <= 120 else args[:117] + "..."
            self._append_component(Spacer(1))
            label = self._append_text(f"{_ACCENT(name)}  {_DIM('running')}  {_MUTED(summary)}")
            self._tool_labels[tool_id] = label
        elif kind == "tool_execution_update":
            tool_id = getattr(event, "tool_call_id", "")
            preview = _tool_preview(getattr(event, "partial_result", None))
            if preview:
                output = self._tool_outputs.get(tool_id)
                if output is None:
                    self._tool_outputs[tool_id] = self._append_text(_DIM(preview))
                else:
                    output.set_text(_DIM(preview))
                    self.tui.request_render()
        elif kind == "tool_execution_end":
            tool_id = getattr(event, "tool_call_id", "")
            label = self._tool_labels.pop(tool_id, None)
            if label is not None:
                state = "failed" if getattr(event, "is_error", False) else "done"
                color = _ERROR if state == "failed" else _SUCCESS
                label.set_text(f"{_ACCENT(getattr(event, 'tool_name', 'tool'))}  {color(state)}")
                preview = _tool_preview(getattr(event, "result", None))
                output = self._tool_outputs.pop(tool_id, None)
                if preview and output is None:
                    self._append_text(_DIM(preview))
                elif preview and output is not None:
                    output.set_text(_DIM(preview))
                self.tui.request_render()

    async def run(self) -> None:
        unsubscribe = self.session.subscribe(self._on_event)
        remove_input = self.tui.add_input_listener(self._on_input)
        try:
            self.tui.start()
            await self.closed.wait()
            if self._prompt_task is not None:
                await self._prompt_task
        finally:
            remove_input()
            unsubscribe()
            self.tui.stop()


async def run_interactive(session: AgentSession) -> None:
    await InteractiveMode(session).run()


__all__ = ["InteractiveMode", "run_interactive"]
