# pi-agent-core (Python port)

Python 复刻版 of [`@earendil-works/pi-agent-core`](../../../packages/agent)（TypeScript 原版位于本仓库 `packages/agent`）。

Stateful agent with tool execution and event streaming：拥有会话转录、生命周期事件流、工具执行（串行/并行）、steering / follow-up 队列，以及可插拔的 LLM provider 层。此外完整移植了 `harness/**`：持久化 lane 运行时（`AgentHarness`）、JSONL 会话存储、压缩/分支摘要、事件总线与钩子注册表。

## 仓库结构

```
py/                       ← uv workspace 根（一份 uv.lock + 一个 .venv）
├── pi-ai/                ← 本 port 依赖的 LLM 层（pi-ai 的依赖面，非全量复刻）
├── pi-agent-core/        ← 本包：agent 循环 + harness 运行时 + 标准工具库
└── pi-simple-cli/        ← `pi` 命令：可执行 demo CLI
```

三个包是同一个 [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) 的成员：
`py/` 下只有一份 `uv.lock` 和一个 `.venv`，依赖统一解析。`pi-agent-core` 通过
`[tool.uv.sources]` 的 `pi-ai = { workspace = true }` 依赖 `pi-ai`（成员间依赖默认
editable，源码改动立刻生效）。

## 安装（uv）

本项目用 [uv](https://docs.astral.sh/uv/) 管理：依赖声明在各包 `pyproject.toml`，
workspace 的解析结果锁在 `py/uv.lock`，虚拟环境是 `py/.venv/`。

```bash
cd py
uv sync                   # 建共享 .venv、装全部成员与 dev 组（幂等）
```

`uv sync` 是幂等的，改了 `pyproject.toml` 后重跑即可；也可以直接跑 `uv run`——它会自动先同步。

关键点：**所有命令都加 `uv run` 前缀**，uv 会自动用 `.venv` 的解释器，不需要 `source .venv/bin/activate`
（想手动激活也可以，见下方「uv 速成」）。

Python 版本由 `.python-version` 固定为 3.12（`requires-python = ">=3.12"`）。运行时依赖只有 `httpx`
（加上 workspace 内的 `pi-ai`）；测试依赖 `pytest` + `pytest-asyncio` 在 `[dependency-groups]` 的 `dev` 组里。

## uv 速成

uv 一个工具顶替 `pip` + `venv` + `virtualenv` + `pip-tools` + `pipx` + `pyenv`（部分）。它用 Rust 写的，
装包和建环境都比 pip 快一个数量级。核心心智模型只有两条：

1. **`pyproject.toml` 是唯一的事实来源**（你要什么），**`uv.lock` 是锁定结果**（实际装了什么版本），
   **`.venv/` 是扔得掉的产物**（可以随时删了重建）。
2. **命令加 `uv run` 前缀**就不用管虚拟环境——uv 自动找到/创建 `.venv`、自动同步依赖、再执行。

### 日常命令

| 你想做的事 | 命令 |
| --- | --- |
| 建环境 + 装全依赖（含 dev） | `uv sync` |
| 只装运行时依赖 | `uv sync --no-dev` |
| 跑命令（自动同步） | `uv run <命令>`，如 `uv run pytest`、`uv run pi --self-test` |
| 开一个 REPL / 脚本 | `uv run python`、`uv run python script.py` |
| 加一个依赖 | `uv add rich` |
| 只加开发依赖 | `uv add --dev pytest-cov` |
| 删依赖 | `uv remove rich` |
| 升级某个包 | `uv lock --upgrade-package httpx` |
| 升级全部 | `uv lock --upgrade` |
| 看看装了什么 | `uv tree` |
| 导出给 pip 用的 requirements | `uv export --no-dev --no-hashes -o requirements.txt` |
| 构建 wheel/sdist | `uv build` |
| 临时跑个一次性命令（不装进项目） | `uvx ruff check .` |

### 只改 `pyproject.toml` 之后

手改 `pyproject.toml`（比如换版本号）后，重跑 `uv sync` 就会重新解析并更新 `uv.lock`。

### CI / 别人拿到这个仓库

```bash
uv sync --frozen     # 严格按 uv.lock 装，不重新解析；锁文件与 pyproject 不一致就直接报错
```

`--frozen` 是 CI 该用的模式：保证每台机器装出完全一样的版本。

### 换 Python 版本

```bash
uv python list              # 看有哪些可用版本（含未下载的）
uv python install 3.13      # 下载一个
uv python pin 3.13          # 改写 .python-version，之后 uv 都用 3.13
uv run python --version     # 确认
```

uv 会自己下载并管理 Python，不依赖 pyenv。

### 想要手动激活虚拟环境

不是必须，但习惯旧流程的话：

```bash
source .venv/bin/activate     # 之后可以不打 uv run 前缀
deactivate                    # 退出
```

### 常见坑

- **别用裸 `pip install`**。裸 pip 装进的是系统/pyenv 环境，不是 `.venv`，会和 `uv run` 看到的两套包。
  要装东西就 `uv add`，要跑就 `uv run`。
- **`uv.lock` 要提交进 git**，`.venv/` 不要（本仓库的 `.gitignore` 已经处理）。
- **不需要手动建 venv**。`uv sync` / `uv run` 会按需创建。
- 报 `ModuleNotFoundError` 时，先确认你是用 `uv run` 跑的；如果是从别的目录跑，加 `--project /path/to/py`。

## 运行（可执行程序）

可执行 demo CLI 已拆到独立包 [`py/pi-simple-cli`](../pi-simple-cli)（`pi` 命令）：

```bash
cd py
uv run pi --self-test          # 离线自测：流式 + 工具调用 + 真实文件系统工具 + 完整 harness 运行时 + 会话重开
uv run pi --prompt "hi"        # faux 模型单次提问；完整参数与 OpenAI 兼容端点用法见该包 README
```

## 库用法（与 TS 版一对一对应）

裸 agent loop：

```python
import asyncio
from pi_agent_core import Agent, AgentOptions, AgentInitialState
from pi_ai.models import create_models
from pi_ai.providers.anthropic import anthropic_provider

models = create_models()
models.set_provider(anthropic_provider())
model = models.get_model("anthropic", "claude-sonnet-4-5")

agent = Agent(AgentOptions(
    stream_fn=models.stream_simple,
    initial_state=AgentInitialState(
        system_prompt="You are a helpful assistant.",
        model=model,
    ),
))

async def main():
    def listener(event, signal):
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)
    agent.subscribe(listener)
    await agent.prompt("Hello!")

asyncio.run(main())
```

持久化 harness 运行时：

```python
import asyncio
from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_ai.models import create_models
from pi_ai.providers.anthropic import anthropic_provider
from pi_agent_core.harness.agent_harness import AgentHarnessOptions, create_agent_harness
from pi_agent_core.harness.env.local import create_local_execution_env
from pi_agent_core.harness.session.jsonl import (
    JsonlSessionCreateOptions, JsonlSessionRepo, JsonlSessionRepoOptions,
)
from pi_agent_core.harness.tool_adapter import StaticToolContext, default_agent_harness_tools

async def main():
    context = BACKGROUND_CONTEXT
    models = create_models()
    models.set_provider(anthropic_provider())
    model = models.get_model("anthropic", "claude-sonnet-4-5")

    env = create_local_execution_env(cwd="/tmp/work")
    repo = JsonlSessionRepo(
        JsonlSessionRepoOptions(file_system=env, sessions_root="/tmp/sessions")
    )
    session = await repo.create(JsonlSessionCreateOptions(cwd="/tmp/work", id="s1"), context)

    result = await create_agent_harness(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=model,
            system_prompt="You are a helpful assistant.",
            tools=default_agent_harness_tools(),
            tool_context=StaticToolContext(env),
        ),
        context,
    )
    harness = result["harness"]
    print("open operations to resume:", result["open"])

    harness.events.on("run_end", lambda event, ctx: print("run", event.status))
    lane = await harness.lane("main", context)
    run = await lane.prompt("hello", context)
    print(run.value.status, run.value.tip_id)

    await harness.close(context)

asyncio.run(main())
```

## API 对照（TS → Python）

### 核心层

| TypeScript (`pi-agent-core`) | Python (`pi_agent_core`) |
| --- | --- |
| `new Agent({ streamFn, initialState })` | `Agent(AgentOptions(stream_fn=..., initial_state=AgentInitialState(...)))` |
| `agent.prompt("hi")` | `await agent.prompt("hi")` |
| `agent.continue()` | `await agent.continue_()`（`continue` 是关键字） |
| `agent.subscribe(cb)` | `agent.subscribe(cb)`（cb 可为 sync/async） |
| `agentLoop(...) / agentLoopContinue(...)` | `agent_loop(...) / agent_loop_continue(...)` |
| `EventStream` / `AssistantMessageEventStream` | 同名；`await stream.result()` → `await stream.result`（property） |
| `setDefaultStreamFn` / `getDefaultStreamFn` | `set_default_stream_fn` / `get_default_stream_fn` |
| `uuidv7()` | `uuidv7()` |
| 事件 `{ type: "agent_start" }` | dataclass `AgentStartEvent`，`event.type == "agent_start"` |
| 工具 `execute(toolCallId, params, signal, onUpdate)` | 同签名（async） |
| `validateToolArguments` | `validate_tool_arguments`（JSON Schema 语义 + 类型矫正） |

### Harness 编排层

| TypeScript | Python |
| --- | --- |
| `AgentHarness.create(options, ctx)` | `await create_agent_harness(options, ctx)`；`AgentHarnessNamespace.create(...)` 为同名入口 |
| `harness.lane(name, ctx)` | `await harness.lane(name, ctx)` |
| `lane.prompt(...)` / `lane.steer(...)` / `lane.followUp(...)` | `await lane.prompt(...)` / `steer(...)` / `follow_up(...)`（同签名） |
| `lane.compact()` / `lane.navigateTree()` / `lane.resume()` / `lane.abort()` | `await lane.compact()` / `navigate_tree()` / `resume()` / `abort()` |
| `lane.watch(ctx)` → `{ snapshot, resnapshot, unsubscribe }` | 同构；`await watcher.resnapshot(ctx)` |
| `lane.inspectExecution(ctx)` | `await lane.inspect_execution(ctx)`（返回 camelCase dict） |
| `harness.getTools()/setTools()/getResources()/...` | `get_tools()/set_tools()/get_resources()/...`（全部 snake_case） |
| `harness.events.on(type, cb)` | `harness.events.on(type, cb)`；事件为 `HarnessEvent` dataclass |
| `HarnessEventBus.emit(event, ctx)` | 同名；payload 可为 wire dict，`harness_event()` 归一化为 `HarnessEvent` |
| `Result.ok / Result.err` | `ok(...) / err(...)`；`result.ok`、`result.value`、`result.error` |
| `new LaneBusy({...})` 等 tagged error | 同名 dataclass + `tag` 属性 |
| 操作状态联合（13 个叶子，`at` 判别） | 13 个 dataclass，`at: str` 判别字段；`revive_operation_state()` 从 JSONL 复原 |
| `RunSettings` / `OperationMeta` / `OperationIntent` | 同名 dataclass（`to_json()` 输出 camelCase 线格式） |
| `CommitDecision` / `FinishDecision` / `LaneReturn` / `LaneReject` | 同名 dataclass |

### 模块映射

| TS 源文件 | Python 模块 |
| --- | --- |
| `src/types.ts` | `pi_agent_core/types.py` |
| `src/agent.ts` | `pi_agent_core/agent.py` |
| `src/agent-loop.ts` | `pi_agent_core/agent_loop.py` |
| `src/stream-fn.ts` | `pi_agent_core/stream_fn.py` |
| `src/proxy.ts` | `pi_agent_core/proxy.py` |
| `@earendil-works/pi-ai`（types/models/utils） | 独立包 [`py/pi-ai`](../pi-ai) → `pi_ai/`（types, models, event_stream, transcript, assistant_message_frame, overflow, text, validation, json_parse, uuid_utils, abort, utils/） |
| `pi-ai providers`（faux / anthropic / openai-completions） | [`py/pi-ai`](../pi-ai) → `pi_ai/providers/` |
| `harness/agent-harness.ts` | `harness/agent_harness.py` |
| `harness/runtime/lane.ts` | `harness/runtime/lane.py` |
| `harness/runtime/harness.ts` | `harness/runtime/harness.py` |
| `harness/runtime/drive.ts` + `harness/runtime/drive/*.ts` | `harness/runtime/drive/__init__.py` + `harness/runtime/drive/*.py` |
| `harness/runtime/{restore,reducer,transcript,progress,types}.ts` | 同名 snake_case 模块 |
| `harness/session/types.ts`（含 13 个 operation 叶子） | `harness/session/types.py` |
| `harness/session/jsonl/{codec,io,io-helpers,storage,repo,fork,legacy-v3,types}.ts` | `harness/session/jsonl/*.py` |
| `harness/session/testing/**`（一致性套件） | `harness/session/testing/**` |
| `harness/compaction/{compaction,branch-summarization,utils}.ts` | `harness/compaction/*.py` |
| `harness/tools/{bash,read,write,edit,edit-diff,path-utils,file-mutation-queue,image}.ts` | `harness/tools/*.py` |
| `harness/env/nodejs.ts`（`NodeExecutionEnv`） | `harness/env/local.py`（同一宿主原生实现对位；直接驱动文件系统与子进程） |
| `harness/{events,hooks,config,skills,system-prompt,messages,prompt-templates,telemetry,result,context,types}.ts` | 同名 snake_case 模块 |

## 测试

```bash
cd py/pi-agent-core
uv run pytest -q
```

当前 **388 passed**（16 个测试文件；CLI 相关测试在 [`pi-simple-cli`](../pi-simple-cli)）。覆盖：

- `test_agent.py` / `test_agent_loop.py`：agent loop 事件序列、transformContext → convertToLlm 管道、工具校验/prepareArguments/before/afterToolCall、并行工具完成顺序与源序持久化、length 截断工具调用的失败处理、steering/follow-up 队列、错误与中止、工具增删声明
- `test_harness_runtime.py`：事件总线投递顺序与失败隔离、lane 恢复、`AgentHarness.create` 装配、lane 创建的幂等与持久化
- `test_harness_e2e.py`：**端到端运行**——faux provider 上跑通完整 harness 运行时：prompt 落盘、事件序列、steering 边界消费、abort、工具批执行与结果安置、JSONL 重开后 tip 与转录一致、deferred 挂起后重开被报告为可恢复的 open operation 且 `resume()` 推进一次 poll、流式中途抓取 lane 快照并归约已提交帧前缀
- `test_harness_session_testing.py`：**一致性套件**——53 个来自 TS `harness/session/testing/**` 的一致性用例，各自在内存与 JSONL 两种后端上各跑一遍（共 106 个参数化测试），覆盖存储契约、会话仓库契约与流式 fork
- `test_harness_jsonl*.py`：JSONL 事务序列化/回放/撕裂行修复、仓库 create/list/open/delete/fork、legacy v3 迁移
- `test_harness_compaction.py`、`test_harness_tools.py`、`test_harness_skills.py`、`test_harness_hooks.py`、`test_harness_misc.py`
- `test_harness_runtime_tools.py`：工具批过程与结果安置
- `test_harness_durable_roundtrip.py`：**durable 值回放保真与跨实现互操作**——13 个 operation 状态叶子 + `OperationMeta`/`OperationIntent` + pending entry + usage row，全部经真实 JSONL 线格式序列化再经真实 reviver 读回，断言复原结果与写入对象相等、且嵌套的 `LaneConfiguration` 仍是 dataclass；另含**读方向互操作**与 **committed write 线格式**测试：用手写的 TS 版线格式行（扁平 entry/usage、`set`/`delete` value、durable operation state、多笔事务）验证 Python 能原样解析，并对 6 种 committed write 逐一断言键集与 TS 声明完全一致（`delete` 无 payload 键、`set`/`append` 必有）
- `test_harness_pico3.py`：pico3 的行为测试 49 项——`Session`/`TxImpl` 的能力校验、作用域、读后写 poison 与 abort 路径，一次走完整 scheduler 的真实回合（generation → tool → post_tools → successor），以及存储提交/回放/撕裂尾修复、文档折叠与 `doc_as_of`、追踪文档 op、view 信封与 head-cut splice、钩子过滤、系统段落折叠、frame 应用、job/plugin kind 阶段与中止

测试只用 faux provider，不触网、不需要真实 API key。

## 状态与范围

移植进度（对照 `packages/agent/src`）：

| 维度 | 已移植 | 说明 |
| --- | --- | --- |
| 模块 | **116 / 116 (100%)** | `packages/agent/src` 下每个 TS 模块都有 Python 对位模块 |
| 代码行 | **33,353 / 33,353 (100%)** | |
| Python 侧 | 147 文件 / 约 49,900 行（本包 128 + [`pi-ai`](../pi-ai) 19） | 测试 18 文件 / 约 10,200 行（另有 [`pi-simple-cli`](../pi-simple-cli) 的 CLI 测试） |

计数口径：TypeScript 模块含 barrel（`index.ts`）与类型文件；Python 侧以对应包
`__init__.py` 的 `__all__` 表达 barrel，以 `harness/types.py` 表达纯类型文件，
**边界情况**：`packages/ai/src`（独立的 provider 库，约 55K 行）只移植了本 port 依赖面上的部分（`types`、`models`、`event-stream`、`transcript`、`assistant-message-frame`、`overflow`、`validation`、`text`、`abort`、`uuid`、`utils`，以及 faux / anthropic / openai-completions 三个 provider），这部分是对位的独立包 [`py/pi-ai`](../pi-ai)，由本包以 workspace 依赖。其余 pi-ai provider 与 API 实现不在本 port 范围内。

已完成（与 TS 功能一一对应）：

- **核心层**：AgentMessage / Message / Usage / Tool 数据模型与 JSON 序列化；AgentLoop（steering / follow-up / prepareNextTurn / shouldStopAfterTurn / 串并行工具执行 / 截断失败处理 / 工具增删声明）；Agent（状态、事件、steer/followUp 队列、abort、reset、continue）；EventStream / AssistantMessageEventStream；JSON-Schema 工具参数校验与类型矫正（TypeBox 语义子集）；Models 注册表与 API key 解析
- **Providers**：faux（脚本化响应、增量流、usage 估算、deferred）、Anthropic Messages、OpenAI chat completions（SSE）
- **运行时**：`harness/runtime/lane.py`（序列化 mutation line、operation 接纳/取消协议、drive claim 循环、配置与队列访问器）、`harness/runtime/harness.py`（session/models/hooks/event bus/全局配置）、`harness/runtime/drive/**`（13 个 operation 状态叶子的完整驱动过程：boundary、checkpoint、generation、tools、tool-placement、response、deferred、reconcile、recovery、structural、retry、terminal）
- **会话**：`harness/session/**`（values、commit、in-memory-storage-state、fork-policy、mutation-line、session、memory、context）、`session/jsonl/**`（v4 事务读写/回放/撕裂行修复、repo、fork、legacy v3 迁移）
- **压缩**：`harness/compaction/**`（compaction、branch-summarization、utils），含阈值压缩、溢出压缩、导航摘要
- **工具**：bash / read / write / edit（含 edit-diff 模糊匹配引擎）/ path-utils / file-mutation-queue / image；串行与并行批执行，源序结果安置
- **编排**：事件总线（按类型投递、失败隔离、缓冲 watcher、resnapshot）、钩子注册表（11 个钩子的聚合语义 + streamOptions patch）、配置校验、effect gate 取消门、工具执行流水线
- **观测**：`harness/telemetry.py`、lane 快照 reducer、`harness/progress.py` 帧进度通道
- **测试基建**：`harness/session/testing/**`（类型、存储装饰器、instrumented/gating storage、benchmark 数据集、conformance 驱动）
- **pico3**：`harness/pico3/**` 全部 26 个模块（约 12,800 行）——`TxImpl`/`Session` 事务内核、`Harness` 编排、`chord` 服务与 view 桥、`scheduler`、`binder`/`membrane`/`view` 文档层、`kinds/{entries,frames,job,plugin,task-api,generation,tool,collapse,post-tools}`

**顶层导入面**：TS 根 barrel `index.ts` 暴露 118 个名字；`pi_agent_core` 顶层覆盖其中 99 个（
`import pi_agent_core as pi; pi.CompactionSettings` / `pi.TelemetryContext` / `pi.generate_summary_with_usage`
均可直接使用），共 322 个顶层导出。余下 19 个中，17 个是 TypeScript 类型层构造（`Infer*Attributes`、
`TelemetrySchemaSpan*`、`*ErrorCode`、`ShellOutputRetention`、`AgentHarnessToolContextSource`）——
在运行时本就不存在实体，Python 侧无对应义务；另外 2 个（`createTypedSpanStarter`、`defineTelemetrySchema`）
来自外部依赖 `@earendil-works/pi-telemetry`，不在本 port 的源码范围内。

**线格式兼容性**：JSONL 事务行与 TS 版逐字节对齐——committed entry/usage write 是「记录本身 + `kind` 标签」的扁平对象，value write 的 `set`/`append` 带 `value` 键而 `delete` 不带，durable 值递归调用各自的 `to_json()`。Python 写出的会话文件可被 TS 版读取，反之亦然。

验证过的端到端行为（`test_harness_e2e.py` + `uv run pi --runtime harness` 实测）：

- prompt → run 落盘 → 事件序列 → 完成记录（含 `fromTipId`/`tipId`/`endedAt`）
- steering 在下一个 boundary 被消费并写入 Branch
- abort 把 operation 收敛为 `aborted` 且保留 tip
- 工具批：写入 → 读取 → 结果按源序安置 → 工具参数暂存清理
- JSONL 会话重开后 tip 与转录一致；`--resume` 可继续
- deferred 响应让 run 持久化挂起（`status: "suspended"`）；重开后 `create` 报告该 operation 为 open，`resume()` 驱动一次持久化 poll 并让挂起状态前进（poll 计数 +1，operation 仍可继续恢复）
- 流式响应期间 assistant 帧按 delta 持久化追加，`lane.watch()` 快照报告由已提交帧前缀归约出的 partial（每一步都是最终响应的前缀）
- `AgentHarness.create` 报告未完成的 open operation（lane / operationId / kind / startedAt）

尚未移植：

- `packages/agent/src` 已无未移植模块。唯一没有 Python 表达的是 TypeScript 的**编译期专用名**，
  它们在运行时本就不存在实体：`Assert`、`IsJson`、`CheckpointAt`、`ConfigOf`、`ConfigOfKinds`、
  `ConfigFor`、`DisjointConfig`、`EntryInput`、`HooksOf`、`InputOf`、`SlotOf`、`TaskOf`。
- `packages/ai/src` 中未被本 port 依赖的 provider/API 实现（如 `api/openai-codex-responses.ts`、
  `auth/oauth/**`、`api/google-*`、`api/bedrock-*`、`api/mistral-*` 等）不在本 port 范围内；参见上方「边界情况」。

未实现的切片会**显式抛错**（`SliceNotImplemented` / `NotImplementedError`），不会静默降级或返回伪造结果。
