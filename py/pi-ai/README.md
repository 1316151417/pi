# pi-ai (Python port)

Python 复刻版 of [`@earendil-works/pi-ai`](../../packages/ai)。当前正从 agent 的依赖面扩展为完整移植，
尚未完成全量复刻。整体顺序与剩余项见 [`PORTING.md`](../PORTING.md)。

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
| `src/api/anthropic-messages.ts` | `pi_ai/api/anthropic_messages.py`；旧 provider 入口保留 |
| `src/api/openai-completions.ts` | `pi_ai/api/openai_completions.py`；旧 provider 入口保留 |
| `src/utils/{headers,sanitize-unicode,hash,error-body,estimate,diagnostics,provider-env,sleep,abort-signals,pi-user-agent}.ts` | `pi_ai/utils/` 下对应 snake_case 模块 |
| `src/utils/typebox-helpers.ts`、`src/session-resources.ts` | `pi_ai/utils/typebox_helpers.py`、`pi_ai/session_resources.py` |
| `src/api/{simple-options,constrained-sampling,transform-messages,openai-prompt-cache}.ts` | `pi_ai/api/` 下对应模块 |
| `src/api/{github-copilot-headers,cloudflare,cloudflare-ai-binding}.ts` | `pi_ai/api/` 下对应模块 |
| `src/auth/{types,context,credential-store,helpers,resolve}.ts` | `pi_ai/auth/` 下对应模块 |
| `src/auth/oauth/pkce.ts` | `pi_ai/auth/oauth/pkce.py` |
| `src/auth/oauth/{device-code,oauth-page}.ts` | `pi_ai/auth/oauth/device_code.py`、`oauth_page.py` |
| `src/compat/extension-oauth-types.ts` | `pi_ai/compat/extension_oauth_types.py` |
| `src/models-store.ts`、`src/model-catalog.ts` | `pi_ai/models_store.py`、`model_catalog.py` |
| `src/models.generated.ts`、`providers/*.models.ts` | `models_generated.py`、`providers/*_models.py`、`providers/data/` |
| `src/auth/oauth/{anthropic,openai-codex,openrouter,radius,github-copilot,kimi-coding,xai,load}.ts` | `auth/oauth/` 下对应模块及共用 HTTP/WHATWG URL 适配 |
| `src/providers/radius-config.ts` | `providers/radius_config.py` |
| `src/providers/{cloudflare-auth,cloudflare-stream,opencode-headers}.ts` | `providers/` 下对应模块 |
| `src/utils/node-http-proxy.ts` | `utils/node_http_proxy.py` |
| `src/api/openai-responses-shared.ts` | `api/openai_responses_shared.py` |
| `src/image-models.generated.ts` | `image_models_generated.py`、`image_models_data.json` |
| `src/{images-models,images-api-registry,images}.ts` | `images_models.py`、`images_api_registry.py`、`images.py` |
| `src/api/{openrouter-images,openrouter-images.lazy}.ts`、`src/providers/openrouter-images.ts` | `api/openrouter_images.py`、`api/openrouter_images_lazy.py`、`providers/openrouter_images.py` |
| `src/api/{google-shared,lazy}.ts` | `api/google_shared.py`、`api/lazy.py` |
| `src/api/{openai-completions,openai-responses,azure-openai-responses}.ts` 及 `.lazy.ts` | `api/` 下对应模块；旧 `providers/openai_completions.py` 入口保留 |
| `src/api/google-generative-ai.ts` 及 `.lazy.ts` | `api/google_generative_ai.py`、`api/google_generative_ai_lazy.py` |
| `src/api/pi-messages.ts` 及 `.lazy.ts` | `api/pi_messages.py`、`api/pi_messages_lazy.py` |
| `src/{compat,cli,oauth,bun-oauth,env-api-keys,image-models,legacy-api-aliases}.ts` | 对应 snake_case 模块及 `compat/` |
| `src/providers/all.ts` | `providers/all.py`（完整模型目录及选定的运行时工厂） |

**接入范围已调整**：用户只要求有代表性的厂商。OpenAI、Azure、OpenRouter 图片和 Faux 已写入，
Anthropic、Google 的完整请求链路已写入；Bedrock、Mistral、Codex 等其余专用接口不作为本轮必做。
通用核心和上层包仍按原目标复刻；旧简化 provider 入口不代表新增模块已通过验证。
`Models` 已接入新鉴权核心，包括凭据、离线恢复、刷新发布和持久化、刷新取消、请求转换和 deferred。
`ImagesModels` 和 OpenRouter 图片生成已写入，共用 SDK HTTP/SSE 传输细节继续补齐。
设备码轮询及全部七个 provider OAuth 流程已写入，包括浏览器回调、手动输入竞争、PKCE、刷新和取消。
内存模型存储通过 deepcopy 隔离读写状态，保留 ETag 字面值；自定义对象复制遵循 Python 协议。
模型类型补齐分级成本、图片输入输出和请求契约；`model_from_json` 保留未知字段和嵌套共享引用，
缺失字段不以默认成本填充。模型目录含 39 个提供方、1354 个定义，来自同版本 npm 发布包，
原生成 JSON 快照在本 checkout 中缺失；来源和哈希记录在 `providers/data/PROVENANCE.md`。
OAuth HTTP 适配使用 `ada-url==4.0.0` 处理 WHATWG URL，配合现有 `httpx==0.28.1`；
重定向、跨 origin 鉴权头、设备码、超时和取消按源码实现，尚未执行登录或联网验证。
54 个图片模型直接来自受版本控制的 TS 目录，生成文件记录源码 SHA-256。
Responses 共用层包含消息/工具转换、工具搜索锚点、grammar 工具、流事件、Azure 推理签名回填、
usage 与停止原因；具体 OpenAI/Azure HTTP provider 已写入，未实现的专用接口按厂商范围处理。
回调的 Python `None` 与 `UNDEFINED` 表示保持 payload，`JSON_NULL` 表示显式替换为 JSON null；
共享 JSON 序列化处理这些哨兵，避免丢失显式 null 能力。
事件流已改为 FIFO 等待者，保留 `await stream.result`；流式 JSON 按 partial-json 0.1.7 原生移植。
Python loader 使用模块顶层导入，按调用延迟执行 API 和 OAuth 操作；不模拟 JavaScript 的动态模块载入时机。
Google 共用层保留思考级别、工具调用签名、多模态工具结果和严格工具 Schema。
OpenAI 原生 SDK 传输层支持 SSE、分块 UTF-8、CR/LF、取消、一次性读取和 tee。Python 显式保留
迭代器后提前退出时须调用 `aclose()`；临时迭代器释放会取消并调度关闭。运行时请求头使用 Python 标识。
Anthropic 默认凭据链补齐 profile/env 解析、OIDC、user OAuth、120/30 秒令牌刷新和失败退避、
凭据文件安全检查及原子写回；没有读取用户凭据、发出请求或验证这些流程。
Anthropic Messages 原生 HTTP/SSE 已接入默认凭据链。Google 使用原生 SDK 传输与转换，
复刻已固定版本的 SSE、请求配置、显式重试和 callable tools；Vertex ADC 不在选定接入范围。
CLI 入口为 `pi-ai` 或 `python -m pi_ai.cli`；仅在实际执行登录时读写 `auth.json`。
Faux 保留既有 `create_faux_core` 的 `(handle, streams)` 返回形式，新增完整 `faux_provider`；
随机数来自 Python PRNG，base-36 文本按 V8 算法转换，许可随包保留。

新增实现当前未写或运行测试；下方测试数量仅记录此前已有版本，不能用于说明本轮改动已通过验证。

## 传输层的经验能力

- **超时**：`SimpleStreamOptions.timeout_ms` 作为每次尝试的传输预算；未设置时不给 httpx 设默认值，
  避免 httpx 自带的 5 秒默认值切断长生成（取消只走请求的 AbortSignal）。
- **传输级重试**：`retry_provider_request` 复刻 OpenAI/Anthropic SDK 的策略——`x-should-retry` 响应头
  优先；无状态码的传输失败重试；否则仅 408/409/429/5xx 重试。延迟取 `retry-after-ms` → `retry-after`
  （秒或 HTTP 日期）→ 指数退避加抖动（`min(0.5*2^i, 8s)` 下浮至多 25%）。服务端要求的延迟超过
  `max_retry_delay_ms`（默认 60s）直接失败而不是干等；退避睡眠可被 AbortSignal 中断。
- **Assistant 调用重试**：`pi_ai/utils/retry.py` 的 `retry_assistant_call`，含账单/配额类错误不重试的经验表。

## 安装 / 使用

本包是 `py/` uv workspace 的成员（与 `pi-agent-core`、`pi-simple-cli` 共用一份
`uv.lock` 和一个 `.venv`）：

```bash
cd py
uv sync              # 一次同步整个 workspace
cd pi-ai
uv run pytest -q
```

`pi-agent-core` 与 `pi-simple-cli` 通过 `[tool.uv.sources]` 的 `{ workspace = true }`
依赖本包（workspace 成员间依赖默认 editable），所以改这里立刻生效。

## 测试

`uv run pytest -q` → **51 passed**。
- `test_pi_ai_overflow_frames.py`：溢出检测/截断判定 + 帧编码器往返
- `test_provider_retry.py`：重试分类（按状态码与响应头）、延迟选择（`retry-after-ms`/秒/HTTP 日期/退避抖动）、
  重试循环（重试到成功/不可重试立即抛/预算耗尽/abort 中断退避）、超时接线、以及本地 mock 服务上的
  流式端到端（503 重试后成功、401 不重试、预算耗尽上报错误）
