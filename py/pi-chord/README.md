# pi-chord

`packages/chord/src` 的 Python 复刻，要求 Python 3.12+、`PyICU==2.16.2`。
函数与配置属性使用 snake_case；服务调用、快照和更新保留 TS 的 camelCase JSON 字段。

## 源码对应

| TS 源码 | Python |
| --- | --- |
| `context/index.ts` | `pi_chord/context.py` |
| `delta/index.ts` | `pi_chord/delta.py` |
| `json.ts`, `types.ts` | `pi_chord/json.py`, `pi_chord/types.py` |
| `services/errors.ts`, `handle.ts`, `instances.ts` | `pi_chord/services/errors.py`, `handle.py`, `instances.py` |
| `services/state.ts`, `state-internals.ts` | `pi_chord/services/state.py`, `state_internals.py` |
| `services/wire.ts`, `state-codec.ts` | `pi_chord/services/wire.py`, `state_codec.py` |
| `services/provider.ts`, `consumer.ts`, `loopback.ts` | `pi_chord/services/provider.py`, `consumer.py`, `loopback.py` |
| `facets/host.ts`, `facets/loader.ts` | `pi_chord/facets/host.py`, `facets/loader.py` |
| `api.ts`, `index.ts` | `pi_chord/api.py`, `__init__.py` |
| `node/manifest.ts` | `pi_chord/node/manifest.py` |
| `node/bundle-loader.ts` 的数据、文件和 loader 生命周期 | `pi_chord/node/bundle_loader.py` |
| `node/bundle.ts`, `node/package.ts`, `bundler.ts` | `pi_chord/node/bundle.py`, `package.py`, `pi_chord/bundler.py` |

包含上下文值链与取消、Delta 变更跟踪与流式路径压缩、状态发布和订阅、singleton/keyed 服务、
组件依赖排序、分阶段激活和清理、保留已有服务句柄的热替换。

`UNDEFINED` 与 Python `None` 不同，分别表示 JS undefined 与 JSON null。
状态未初始化时返回 `UNDEFINED`；`apply_immutable` 保留未变更子树并复制修改路径。
Tracker 的字典/数组视图实现 Python 的 MutableMapping/MutableSequence，写入必须经过视图。
内部 decoded Delta 操作使用元组；encoded WireOp 和服务发布边界使用原生 list，满足严格 JSON 校验。

远程实现采用字典成员或实例自身属性，与 TS 的 `Object.keys` 对应。类的继承方法不会自动发布；
可以使用 `{"method": instance.method}` 暴露绑定方法。异步操作需要运行中的 asyncio 事件循环。
状态监听器接收 `(value, context, delivery)` 三个参数。

Bundle 读取保留原始 camelCase JSON、版本 2、可选字段省略、只读返回值、文件名约束、
SHA-256 校验和 source map。字符串参数按文件路径处理；URL 重载接收 urllib 的
`ParseResult` / `SplitResult`。dataclass 的 `to_wire()` 产生原格式 JSON 对象。

`pi_chord.node` 提供 `create_facet_bundle_loader` 和 `create_facet_bundle_artifact_loader`，
分别接收 `FacetBundleLoaderOptions` / `FacetBundleArtifactLoaderOptions`，并且都要求显式传入
仅限关键字参数 `execute_common_js: CommonJsExecutor`。调用方式为
`create_facet_bundle_loader(options, execute_common_js=host_executor)`。
加载器在每次 `load()` 时重新读取 manifest、校验源内容、验证非空 default 导出及重复 facet ID；
artifact 在创建 factory 时先校验，每次加载分配新临时目录，并在失败和 dispose 时清理。
返回的 `facets` 是只读元组，保留 facet 对象身份；dispose 幂等并清空该代 facets。
多个清理错误使用 Python `BaseExceptionGroup`（普通异常自动收窄为 `ExceptionGroup`）聚合。
facet 的 setup、激活和资源释放仍由 facet host 执行，loader 不重复调用这些生命周期钩子。

执行器是同步 callable，接收 `(source, module_path, external_imports, resolve_external)`，
对应原 `executeCommonJsModule` 的完整边界。它负责实际 CommonJS 执行、外部依赖预解析、
包名与目标 URL 校验、受声明列表约束的 `require` / `require.resolve`、Chord 宿主导出映射、
外部模块缓存以及 JS 对象/闭包到 Python 宿主的桥接。resolver 返回 `UNDEFINED` 表示继续默认解析；
`None` 不作为 resolver 的未命中值。每次调用必须产生新的 entry 执行代次。
执行器返回 `module.exports` 对应的 Mapping 或带 `default` 属性的对象；default 是一个或一组
`Facet`，也可使用提供 `id` 和可调用 `setup` 属性的桥接对象。普通 dict facet 必须由执行器桥接，
因为 Python facet host 使用属性访问。运行时引用应跟随导出对象存活，不能由 loader 缓存 entry。
这个必填接口是 Python 适配扩展，不代表已经提供默认 JavaScript/Node 执行引擎。

构建保留原 TS/JS → CommonJS 工具链，Python 直接使用
[esbuild 0.28.2 的原生 service 协议](https://github.com/evanw/esbuild/blob/v0.28.2/lib/shared/stdio_protocol.ts)。
运行构建 API 时需要 esbuild 0.28.2 原生程序，按 `ESBUILD_BINARY_PATH`、祖先目录的
`node_modules/@esbuild/<platform>`、PATH 查找，握手时严格检查版本；没有下载、安装或 Node 代理。
每次 bundle 调用复用一个原生进程，结束后清理。实现含默认 target、源码映射、独立 entry 构建、
内容寻址文件、元数据输出校验、原目录备份替换和失败回滚。本阶段没有实际运行构建。

服务实例和 bundle entry 的 locale 排序使用 ICU；与 Node 使用相同的 ICU 数据版本及默认 locale
是严格一致的前提。运行时原生依赖和 Python 异步取消仍需最后统一验证。

## 尚待完成

- 默认 CommonJS 执行引擎仍未提供；外部模块解析、VM 执行和对象桥接由必填执行适配器负责。
  loader 生命周期已实现，没有提供占位成功执行器，也不会把版本 2 JavaScript artifact 当 Python 源码执行。
- 现有 `pi-agent-core/_chord` 和 pico3 的局部实现尚未迁移到独立包。
- Tracker 写入时按对象身份遍历定位，与 TS 的缓存路径结构存在性能差异。
- InstanceDirectory 使用 live Map 迭代并保留先取消、后移除 task 的顺序；此适配仍待统一验证。
- Python coroutine 启动时机与 JS Promise 不完全相同，生命周期、重入、取消和保留引用行为待统一验证。

本阶段没有新增或运行测试。源码已写入不等于一对一行为已经验证。
