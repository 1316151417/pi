# pi-coding-agent (Python)

This package begins the Python port of `packages/coding-agent/src/core` at
its lowest application layer. `messages.py` maps the coding agent's bash,
custom, branch-summary, and compaction-summary messages to pi-ai messages.
`system_prompt.py` reproduces the ordered prompt sections and their patch
logic; `skills.py` renders pre-loaded skill metadata for that prompt.
`paths.py` and `tools/path_utils.py` cover local path normalization and
read-path fallbacks. `tools/file_mutation_queue.py` serializes writes to the
same canonical path. Shared truncation logic is reused from `pi-agent-core`.
`tools/write.py` provides a concrete tool with pluggable file operations
and an `AgentTool` adapter. `tools/edit.py` reuses the agent-core edit engine
for unique, non-overlapping replacements, preserving BOM and line endings.
Its diff and patch are generated with Python `difflib`, so formatting may
differ from the TypeScript `diff` package. `tools/read.py` implements text pagination,
truncation notices, image detection, and image attachments. Image conversion
and resize use pinned Pillow 11.3.0 in place of the TypeScript Photon/WASM
implementation; image bytes can differ even when dimensions and MIME policy
match.

`tools/bash.py` and `tools/output_accumulator.py` cover local streaming shell
execution, cancellation, timeouts, bounded output, temp-file spill, and tool
updates. `bash_executor.py` handles interactive shell commands. The default
coding tool set is `read`, `bash`, `edit`, and `write`; separate `grep`, `find`,
`ls`, and PowerShell tools are not needed for the core shell workflow.
`session_entries.py` and `session_manager.py` implement the append-only JSONL
session tree, context projection, compaction boundaries, branching, and fork.
`model_config.py` loads immutable custom model configuration. `model_runtime.py`
composes model definitions with the Chat Completions, Responses, and Anthropic
Messages transports; `auth_storage.py` keeps credentials in `auth.json` and
supports runtime overrides. `agent_session.py`
connects the session tree and default tools to `pi-agent-core`, persists agent
events, and exposes prompt, queueing, model, thinking, bash, and compaction operations.
`model_resolver.py` reproduces model matching, scoped selection, and session
restore fallback while keeping defaults for the two representative providers.
`resources.py` finds project context files and skill metadata for the system
prompt. `cli.py` supplies one-shot text/JSON output; interactive terminal runs
through `pi-tui` with a multiline editor, Markdown replies, streamed text,
tool status, session history, model switching, and cancellation (`pi-python`
or `python -m pi_coding_agent`).

From the repository root, start the terminal UI with a configured model:

```sh
uv run --directory py --package pi-coding-agent --frozen pi-python --provider deepseek --model deepseek-flash
```

Enter submits, Shift+Enter or Ctrl+J inserts a newline, Ctrl+C aborts the
current response (or clears the editor), and Ctrl+D or `/exit` exits. `/model`
lists configured models, `/model provider/model-id` switches models, and
`/compact` compacts the current session. Use `--continue` to reopen the most
recent session. `-p "prompt"` keeps one-shot, non-TUI output.

The extension runner, full TypeScript interactive mode, RPC mode, package
manager, full YAML and gitignore parsing for skills, and exact split-turn compaction are outside this
core workflow. The Python package does not yet ship the original
docs/examples trees referenced by the default prompt; `PI_PACKAGE_DIR` can
point at a package directory containing them. The Python workspace passes
syntax compilation and its existing AI, agent-core, and simple CLI suites.
The coding-agent prompt, tool, session persistence, model configuration, and
compaction paths were also exercised with a local faux provider. The terminal
UI was exercised through the real AgentSession event path with the faux
provider; no paid provider request was made.
