# pi-simple-cli (Python port)

`pi` 命令 —— pi-agent-core Python port 的可执行 demo CLI，从 `pi-agent-core`
包中拆出：agent / harness 逻辑在 [`pi-agent-core`](../pi-agent-core)，provider 在
[`pi-ai`](../pi-ai)，本包只保留命令行入口与终端渲染。

## 运行模式

```bash
pi                          # 交互式聊天（faux 离线模型）
pi --prompt "hi"            # 单次提问后退出
pi --self-test              # 离线自测：流式 + 工具调用 + 真实文件系统工具 + 完整 harness 运行时 + 会话重开
pi --provider anthropic     # 真实 Anthropic API（需 ANTHROPIC_API_KEY）
pi --provider openai        # 真实 OpenAI API（需 OPENAI_API_KEY）
pi --provider anthropic --model claude-sonnet-4-5 --thinking high
pi --tools none             # 无工具对话
```

### 接 OpenAI 兼容 API（DeepSeek / vLLM / Ollama / GLM / Qwen…）

凡是说 OpenAI chat-completions 协议的端点都能接：`--base-url` 指过去，`--model` 随便起名，
不在内置目录里也会自动注册。以 DeepSeek 为例：

```bash
export DEEPSEEK_API_KEY=sk-...

uv run pi --provider openai \
  --base-url https://api.deepseek.com \
  --api-key "$DEEPSEEK_API_KEY" \
  --model deepseek-chat \
  --prompt "你好"

# 也可以全用环境变量（OPENAI_BASE_URL 是 --base-url 的环境变量写法）
export OPENAI_API_KEY="$DEEPSEEK_API_KEY" OPENAI_BASE_URL=https://api.deepseek.com
uv run pi --provider openai --model deepseek-reasoner

# 配合持久化 harness 运行时
uv run pi --provider openai --base-url https://api.deepseek.com \
  --model deepseek-chat --runtime harness --session ds1
```

本地模型同理：vLLM（`--base-url http://localhost:8000/v1`）、Ollama（`--base-url http://localhost:11434/v1`）。
`tests/test_custom_openai_endpoint.py` 用本地 SSE mock 服务端到端验证了这条通路（含鉴权头、
流式增量、usage 统计），不依赖真实网络。

### 两种运行时（`--runtime`）

| 运行时 | 说明 |
| --- | --- |
| `--runtime agent`（默认） | 裸 agent loop：`Agent` + `agent_loop`，事件通过 `agent.subscribe` |
| `--runtime harness` | 完整 `AgentHarness` lane 运行时：持久化 operation 状态机、驱动的工具批、压缩、可恢复会话 |

```bash
# JSONL 会话持久化 + 恢复（两种运行时都支持）
uv run pi --session my-session                    # 新建并持续写入会话
uv run pi --resume                                # 继续该目录下最近的会话
uv run pi --sessions-root /path/to/sessions --session s1

# 走完整 harness 运行时：operation 状态机会落盘，可在任意时刻中断后原样恢复
uv run pi --runtime harness --session s1 --prompt "refactor this file"
uv run pi --runtime harness --resume --prompt "keep going"
```

交互模式命令：`/quit` 退出、`/reset` 清空会话（agent 运行时）、`/state` 查看转录或 lane 快照、
`/lanes` 列出 lane 与其 operation、`/abort` 取消当前 operation（harness 运行时）。

## 环境

本包是 `py/` uv workspace 的成员。在 `py/` 下 `uv sync` 一次即可（一份 `uv.lock`、
一个 `.venv`，三个包共用）：

```bash
cd py
uv sync                     # 建共享 .venv、装全部成员与 dev 组
uv run pi --self-test
uv run --package pi-simple-cli pi --help   # 显式指定成员时用 --package
```

## 测试

```bash
cd py/pi-simple-cli
uv run pytest -q
```

当前 **8 passed**（2 个测试文件）。覆盖：

- `test_custom_openai_endpoint.py`：OpenAI 兼容端点通路——`--base-url` / `OPENAI_BASE_URL`
  重定向 provider、未知 model id 注册、真实 Agent 回合端到端（本地 SSE mock，不触网）
- `test_cli_self_test.py`：`--self-test` 覆盖 agent 与 harness 两个运行时
