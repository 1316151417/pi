# pi-ai (Python port)

Python 复刻版 of [`@earendil-works/pi-ai`](../../packages/ai) —— 只覆盖本仓库 Python port 的**依赖面**
（约 19% 的 TS 源码行数），不是 pi-ai 的全量复刻。

## 装了哪些

| TS 源 | Python |
| --- | --- |
| `src/types.ts` | `pi_ai/types.py`（消息、内容块、Usage、Tool、Model、DeferredHandle、事件） |
| `src/models.ts` | `pi_ai/models.py`（Models 注册表、Provider、鉴权、`stream`/`stream_simple`/`complete`/`stream_deferred`） |
| `src/utils/event-stream.ts` | `pi_ai/event_stream.py` |
| `src/utils/transcript.ts` | `pi_ai/transcript.py` |
| `src/utils/assistant-message-frame.ts` | `pi_ai/assistant_message_frame.py` |
| `src/utils/overflow.ts` | `pi_ai/overflow.py` |
| `src/utils/json-parse.ts` | `pi_ai/json_parse.py` |
| `src/utils/validation.ts` | `pi_ai/validation.py` |
| `src/utils/retry.ts` | `pi_ai/utils/retry.py`（assistant 调用的重试策略） |
| `src/utils/provider-retry.ts` | `pi_ai/utils/provider_retry.py`（传输层重试） |
| `src/utils/{text,abort,uuid}.ts` | `pi_ai/{text,abort,uuid_utils}.py` |
| `src/providers/faux.ts` | `pi_ai/providers/faux.py` |
| `src/api/anthropic-messages.ts` | `pi_ai/providers/anthropic.py` |
| `src/api/openai-completions.ts` | `pi_ai/providers/openai_completions.py` |

**未包含**：其余 provider 适配器（Google、Bedrock、Mistral、OpenAI Responses/Codex、pi-messages 等）、
OAuth 流程、完整模型目录与 image models、各厂 compat 自动检测体系。这些不在 pi-agent-core 的依赖面上。

## 传输层的经验能力

- **超时**：`SimpleStreamOptions.timeout_ms` 作为每次尝试的传输预算；未设置时不给 httpx 设默认值，
  避免 httpx 自带的 5 秒默认值切断长生成（取消只走请求的 AbortSignal）。
- **传输级重试**：`retry_provider_request` 复刻 OpenAI/Anthropic SDK 的策略——`x-should-retry` 响应头
  优先；无状态码的传输失败重试；否则仅 408/409/429/5xx 重试。延迟取 `retry-after-ms` → `retry-after`
  （秒或 HTTP 日期）→ 指数退避加抖动（`min(0.5*2^i, 8s)` 下浮至多 25%）。服务端要求的延迟超过
  `max_retry_delay_ms`（默认 60s）直接失败而不是干等；退避睡眠可被 AbortSignal 中断。
- **Assistant 调用重试**：`pi_ai/utils/retry.py` 的 `retry_assistant_call`，含账单/配额类错误不重试的经验表。

## 安装 / 使用

```bash
cd py/pi-ai
uv sync
uv run pytest -q
```

pi-agent-core 通过 `[tool.uv.sources]` 以 editable 路径依赖本包，所以改这里立刻生效。

## 测试

`uv run pytest -q` → **51 passed**。
- `test_pi_ai_overflow_frames.py`：溢出检测/截断判定 + 帧编码器往返
- `test_provider_retry.py`：重试分类（按状态码与响应头）、延迟选择（`retry-after-ms`/秒/HTTP 日期/退避抖动）、
  重试循环（重试到成功/不可重试立即抛/预算耗尽/abort 中断退避）、超时接线、以及本地 mock 服务上的
  流式端到端（503 重试后成功、401 不重试、预算耗尽上报错误）
