# Python 复刻进度

目标：按依赖顺序将仓库 TS 的核心骨架移植到 `py/`，用于阅读、学习和复刻。实现阶段未新增单测、未运行测试；
主链写入后才做统一验证。下列「已写入」与既有测试通过均不等于已证明所有 TS 行为完全等价。

2026-09-27 已恢复工作。新范围判断：实现能解释主调用链的代码；重复厂商对接、模型清单扩张、
非核心边缘实现不继续追逐。模型通信只需 Chat Completions、Anthropic Messages、Responses 三种
协议。已有的 Google、Azure、OpenRouter 等实现保留，但不作为后续复制扩张的理由。

下一步只补这三种协议和 agent/会话/远程调用的必要支撑；已经写入的附加适配保留。

## 当前依赖顺序

1. 独立基础层：`chord`、`telemetry`、`tui`。
2. 依赖基础层：`ai`、`protocol`。
3. 依赖 AI/Chord：`durable`、`agent`，以及 `session-backends/sqlite-node`。
4. 依赖协议和 agent：`client`、`server` 的握手、路由、请求与会话主路径。
5. 上层产品：仅选择有助于理解主流程的 `coding-agent` 入口；`evals` 与演示/文档场景不是骨架必需项。

互不依赖的子模块可以并行；同一依赖链先补被依赖模块。

## 已有实现

- `pi-ai`：消息、事件流、transcript/frame、部分模型注册和鉴权、重试与三个 provider。
  之前只覆盖 agent 的依赖面，不能按全量完成计。
- `pi-agent-core`：agent loop、harness、持久化会话、JSONL、工具、压缩、pico3 等已有实现。
  Chord context 和 telemetry 已接入独立基础包；其余既有实现仍需和当前 TS 逐文件对照。
- `pi-simple-cli`：现有演示命令行，继续保留。

## 本轮已写入

- `pi-telemetry`：契约、schema helper、空实现、内存记录、testing fixture 类型。
  conformance 用例留到统一测试阶段。
- `pi-chord`：context、Delta、JSON 校验、服务类型/错误/线格式、状态编解码与订阅、
  provider/consumer/loopback、facet 生命周期/依赖顺序/热重载、公共入口。
- Chord bundle：manifest/artifact 类型与 JSON、文件读取和完整性校验；package.json 配置与
  路径解析、原生 esbuild service 协议、内容寻址构建、目录替换及回滚。两个 loader factory 的
  每代加载、导出校验、临时目录和失败/释放清理已写入，要求必填 `execute_common_js` 适配器；
  默认 CommonJS 执行、Node 外部模块解析和宿主对象桥接仍未提供，版本 2 artifact 保持 JavaScript。
- `pi-tui`：UTF-16/ANSI/Unicode 显示工具、分词、模糊匹配、按键与键位、自动补全、撤销、
  输入缓冲、终端颜色/图片、原生平台与剪贴板、LaTeX、TUI 基类、终端驱动、布局和主屏渲染。
- TUI 基础组件：Text/Box/Spacer/TruncatedText/Image、Loader/CancellableLoader、
  MouseRegion、Stack/VStack/HStack、ScrollView、Input、SelectList、SettingsList、
  全屏搜索索引和临时提示；Markdown 组件、完整所需 Marked lexer/tokenizer 和扩展。
  多行 Editor、全屏渲染器及 Marked HTML Parser/Renderer/Hooks/扩展/异步 API 已写入并整合公开入口。
- `pi-protocol`：CBOR、分帧、协议 v8 类型、严格消息校验、流式消息解码及公共入口。
- `pi-client`：传输无关连接、握手、请求取消、服务订阅与状态更新、Unix socket 传输与发现；
  已接入 workspace，尚未运行或测试。依赖现有 Chord 与协议包。
- `pi-server`：传输无关服务端、协议握手、RPC 取消、服务订阅快照与增量编码、
  Session 附着/脱离/清理路由及 Unix socket 监听；已接入 workspace 和 uv.lock，尚未运行或测试。
- `pi-coding-agent`：从底层 `core/messages.ts` 开始，已写入 bash、扩展消息及分支/压缩摘要到
  AI 消息的转换；系统提示词的有序分段、差异更新、技能元数据格式化和思考等级默认值已写入。
  路径规范化、读路径回退和同文件写入队列已写入；截断逻辑复用 `pi-agent-core`，不重复实现。
  `write`、`read`、`edit` 与 `bash` 工具、ToolDefinition 到 AgentTool 的主路径适配已写入；`edit`
  复用 agent-core 的替换引擎，diff/patch 由 Python `difflib` 生成，与 TS 格式可能不同。读图的签名识别、
  格式转换和缩放使用 Pillow 11.3.0 对应 TS Photon/WASM 的行为边界，编码字节不保证一致。
  `bash` 的流式输出、取消/超时、截断和交互执行器已写入；JSONL 会话树、压缩上下文投影、
  分支和 fork 的主路径已写入。独立的 `grep`/`find`/`ls`/PowerShell 工具是重复能力，
  不作为核心骨架完成条件。
  不可变的 `models.json` 配置读取、三种协议模型组合、`auth.json` 持久凭据与运行时覆盖、
  AgentSession 的基础提示、消息持久化、队列、模型/思考等级和交互 bash 已写入；
  模型选择、会话恢复回退、项目指令与技能发现、手动/阈值压缩和 CLI 文本/JSON 主路径已写入。
  压缩采用完整用户轮次切点，未覆盖原版拆分单个超大轮次的复杂分支。
  包已接入 workspace 和 uv.lock。
  基础终端 UI 已接入 `pi-tui`：多行输入、流式回复、工具状态、会话恢复、模型切换与取消。
  扩展系统、完整 TS 交互模式/RPC、包管理器与大量边缘命令不作为本次核心骨架完成条件。
- `pi-ai` 工具：headers、Unicode 清理、hash、错误体、token 估算、diagnostics、provider env、
  sleep、合并取消信号、user agent；底层取消信号补齐监听器和取消等待。
- AI 消息 diagnostics 改为 TS 的 `type/timestamp/error/details`，补齐读写线格式。
- AI 请求选项补 telemetry context、fetch、env、WebSocket 连接超时；未指定 transport/cache retention
  交给 provider 解析，和 TS 的 optional 字段对应。
- AI provider 前置层：simple options/思考预算、消息转换、JSON Schema/grammar 约束采样、
  OpenAI cache key、Copilot 请求头、Cloudflare endpoint 与 binding fetch。
- AI 鉴权基础层：完整 auth 类型、默认上下文、按 provider 串行的内存凭据存储、
  API key/lazy OAuth helpers、PKCE、鉴权解析与 OAuth 锁内刷新/超时取消。
- OAuth 共用设备码轮询、HTML 结果页、旧扩展回调契约；模型 catalogue flatten 和 ModelsStore。
- 全部七个 provider OAuth 登录/刷新/取消、浏览器回调、共用 HTTP/WHATWG URL 和 loader 适配。
- 模型完整字段、分级成本、共享 wire 视图、图片和请求契约；39 个提供方的 1354 个模型定义来自
  同版本 npm 发布包，已核对数据完整性。原 checkout 未包含生成 JSON，不能声称快照与 HEAD 完全一致。
- 54 个图片模型由当前受版本控制的 TS 文件转换；ImagesModels、图片 API 注册和 OpenRouter
  图片生成执行链路已写入，SDK HTTP/SSE 传输细节继续补齐。
- Models 已接入新鉴权核心，补齐离线模型恢复、刷新发布与持久化、刷新代次取消、请求转换和
  deferred 调度；lazy API、图片集合及模型集合公开入口已接入。
- Responses 共用消息/工具转换和流事件处理；Cloudflare 鉴权/endpoint 与 OpenCode session header；
  HTTP proxy 解析。未调用任何 provider 服务。
- Google 共用思考预算、多模态消息/工具结果、签名和工具 Schema 转换，以及完整 Generative AI
  参数、流事件、usage、错误处理和 lazy/provider 工厂；原生 SDK HTTP、SSE、转换及重试已写入。
- OpenAI Completions、Responses、Azure Responses 的参数、流事件、错误与 lazy 工厂；
  共用原生 HTTP/SSE（UTF-8、CR/LF、取消、单次消费、tee）已写入。接入 15 个对应工厂。
- Faux 已对齐共享引用、UTF-16 分块、deferred 取回/取消和明确无密钥鉴权；CLI 接入完整工厂。
- Anthropic SDK 默认配置/凭据链、OIDC 和 user OAuth、共享 TokenCache、文件安全及原子写回已写入。
- Anthropic Messages 完整请求转换、thinking、工具、流事件、usage、错误及 lazy/provider 工厂，
  原生 SDK HTTP/SSE 和凭据链整合已写入；OpenAI、Anthropic、Google 为本轮代表性接口。
- AI `providers/all`、图片目录、环境密钥发现、legacy API 别名、compat 全局注册表与流路由、
  OAuth 兼容入口及 `pi-ai` 登录 CLI 已写入；没有运行 CLI 或登录流程。
- PiMessages 内部协议、请求/响应钩子、SSE、rewrite/失败 diagnostics、lazy 工厂及 compat 注册已写入；
  与 Agent proxy 共用消息 wire 视图，保留未知键、显式 null、参数引用及数组洞的序列化。
- `pi-durable`：完整持久化类型和内存 Storage，包括跨表 ID、不可变行校验、索引、fork、分页及复制。
- `pi-session-backend-sqlite`：原生 sqlite3 适配、参数化 SQL、原样 schema migration、行编解码、
  分支索引、Storage 事务/快照队列、Session admission/held mutation/关闭，以及 Repo
  create/open/list/delete/fork/close 已写入。已接入 workspace 和 uv.lock；未建库、运行 SQL 或测试。
  SQLite 版本/异常、prepare 的 EXPLAIN 适配、游标生命周期和 JSON record 类型边界见该包 README。
- Agent 核心源码对照修正：共享引用、工具更新回调调度、listener 集合迭代、continue steering；
  proxy 改用共用 HTTP 和消息 wire 视图，保留未知字段、fractional usage、工具参数引用与稀疏内容。
- Agent Chord context 直接重导出独立包并迁移调用，补齐取消竞速/监听器释放；telemetry 复用
  独立包与完整 AI/Harness schema。会话 mutate 支持任意 Awaitable，列表上限校验按 JS safe integer。
- EventStream FIFO/终止语义、partial-json 0.1.7 全分支和 overflow 正则语义已做源码校正。
- AI validation 按当前 TypeBox 1.3.27 源码重写：普通 JSON Schema 的类型、对象/数组约束、
  组合与条件、依赖、引用作用域、unevaluated 记录、21 个默认格式及错误消息已写入；
  保留可选 null 清理、两个转换阶段、对象身份缓存和默认最多 8 条错误的源码顺序。
  `partial-json` 与 TypeBox 的原始 MIT 许可和源码版本记录已写入 `pi_ai/licenses/`。
- AI session resource cleanup 和 string-enum JSON Schema helper。
- 源码对照修正旧重试实现：错误分类、显式零延迟上限、SDK 异常结构和 Retry-After 解析；
  取消信号区分省略 reason 与显式 `None`，监听器异常不会中断后续通知。
- 工作区注册 `pi-chord`、`pi-telemetry`、`pi-protocol`、`pi-tui`、`pi-durable`；
  `pi-ai` 显式依赖 `pi-telemetry`，agent 显式依赖 Chord/telemetry。
  Chord/TUI 的 ICU 依赖精确固定为 PyICU 2.16.2，锁文件已更新；没有安装或运行 PyICU 构建。

## AI validation 的待补分支与适配边界

普通 JSON Schema 校验是当前实现里程碑，不等于完整移植 TypeBox 的类型构造和类型计算 API，
也不代表已经通过差异测试。Python 的 `validate_tool_call`、`validate_tool_arguments`、
`ValidationError` 和既有 `collect_errors` 入口均保留。

- `Value.Convert` 已实现 JSON 可表示的 `~kind` 原始类型、Object、Record、Array、Tuple、
  Ref/Cyclic、Union、Enum、TemplateLiteral 和常见 Intersect。普通 schema 没有 `~kind` 时，
  该阶段保持源码的直接返回行为，随后由 pi 的 JSON Schema 递归转换处理。
- 暂停回补 `FromBigInt` 与 BigInt Literal：需要独立的 JavaScript BigInt 值模型，不能把
  Python 普通整数同时当作 JS Number 和 BigInt。
- 暂停回补 `FromIntersect -> Instantiate` 的 `Call`、`Rest` 和 `InstantiateDeferred`：
  包括 Add/RemoveImmutable、Add/RemoveReadonly、Add/RemoveOptional、Capitalize、
  Conditional、ConstructorParameters、Evaluate、Exclude、Extract、Index、InstanceType、
  Interface、KeyOf、Lowercase、Mapped、Module、NonNullable、Pick、Parameters、Partial、
  Omit、ReadonlyObject、Record、Required、ReturnType、TemplateLiteral、Uncapitalize、
  Uppercase、With，以及泛型调用、rest 展开和相应推断环境。
- 暂停回补 `EvaluateDependent -> ExcludeOperation`，以及涉及 Function、Constructor、Infer、
  Generic、Deferred 等类型的 `Extends` 比较。当前可达但未实现的类型计算明确抛出
  `UnsupportedTypeBoxSchema`，不把它们作为成功转换或全量完成。
- Python schema dict 不能弱引用，缓存以强引用保留对象身份直到模块生命周期结束；
  JavaScript 的 Symbol 元数据、自定义原型/setter、属性描述符和非 JSON 运行时对象仍存在语言边界。
- 正则按 ECMAScript 语法解析后使用原生 `regex==2025.7.34` 匹配。重复原子内嵌捕获的反向引用、
  lookbehind 内或引用其捕获的反向引用、作用域 `i` 修饰、超过 `2**32-1` 的重复计数、legacy
  畸形 `\\c` 和后端无法表示的结构明确抛出 `UnsupportedRegexError`；`format: regex` 仍区分
  「语法合法但匹配暂不支持」和无效正则。Unicode 属性表固定为 16.0，NFC 规范化随 Python
  运行时 Unicode 数据版本。未调用 Node/JS 代理执行校验。

## 后续实现清单

- 继续对照 TS 交互模式，补齐 UI 细节与扩展入口；对 coding-agent 主链做更全面的差异测试。
- Agent 会话 JSONL/Entry/UsageRow 的缺失与 null、未知字段和数值保真；AI/Chord 信号类型边界。
  `watchSession` 在 TS 原版也直接抛 `SliceNotImplemented`，不是 Python 漏实现。
- Chord CommonJS 宿主执行需外部执行适配器，完整 Node 外部模块执行不是当前主路径必需。
- Bedrock、Mistral、Codex 及其余厂商特有认证/参数不继续复制；`evals` 和大量示例也不作为完成条件。
- 后续若要扩大等价性保证，增加 TS/Python 差异测试；当前核心骨架已实现，但不能宣称所有 TS 行为完全等价。

包内 README 记录源码映射和 Python 语言适配边界。未实施 git commit。

验证结果：`python3 -m compileall -q py` 通过；Python 已有测试按包运行，`pi-ai` 51、
`pi-agent-core` 388、`pi-simple-cli` 8 项通过。使用 faux provider 的临时集成脚本覆盖
coding-agent 提示→回复、写工具、JSONL 会话恢复、模型配置/选择、资源加载与压缩，均通过；
全部 12 个 Python workspace 包可导入；临时脚本已移除。新增终端交互经真实 AgentSession 与
faux provider 离线烟测，确认增量回复、状态和用量显示；未调用付费或真实模型厂商。
`npm run check` 的 Biome、依赖、入口图、shrinkwrap 检查通过；TS 类型检查仍因当前生成模型目录
与若干 TS 测试期待的模型 ID 不一致而失败，未修改无关 TS 源码。
