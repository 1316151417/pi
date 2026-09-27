# pi-tui（Python）

按 `packages/tui/src` 的依赖顺序移植。当前是实现阶段，未新增或运行测试，不能视为已经验证等价。

## 已写入

| TypeScript | Python |
| --- | --- |
| `fuzzy.ts` | `fuzzy.py` |
| `kill-ring.ts` | `kill_ring.py` |
| `undo-stack.ts` | `undo_stack.py` |
| `autocomplete.ts` | `autocomplete.py` |
| `keys.ts` / `keybindings.ts` | `keys.py` / `keybindings.py` |
| `utils.ts` / `word-navigation.ts` | `utils.py` / `word_navigation.py` |
| `stdin-buffer.ts` | `stdin_buffer.py` |
| `terminal-colors.ts` / `terminal-image.ts` | `terminal_colors.py` / `terminal_image.py` |
| `native-module-path.ts` / `native-platform.ts` / `native-modifiers.ts` | 对应 snake_case 模块和 `_native_*` 后端 |
| `latex.ts` | `latex.py` |
| `tui.ts` | `tui.py` / `_component.py` |
| `terminal.ts` | `terminal.py` / `_process_io.py` |
| `layout.ts` / `tui-main-screen.ts` | `layout.py` / `tui_main_screen.py` |
| `layout-node.ts` / `editor-component.ts` | `layout_node.py` / `editor_component.py` |
| `alt-screen-search.ts` | `alt_screen_search.py` |
| `components` 的 Text/Box/Spacer/TruncatedText/Image | `components` 对应模块 |
| Loader/CancellableLoader/AltScreenFlashContainer/MouseRegion | `components` 对应模块 |
| Stack/VStack/HStack/ScrollView/Input/SelectList/SettingsList | `components` 对应模块 |
| `components/markdown.ts` / Marked 18.0.11 | `components/markdown.py` / `_markdown_lexer.py` / `_marked_*` |
| `components/editor.ts` / `tui-alt-screen.ts` | `components/editor.py` / `tui_alt_screen.py` |

TUI 源文件的对应实现及公开 Marked 的 HTML parser/renderer/hooks 等接口已写入。

## Python 适配

- 公开名称用 snake_case，选项和结果用 dataclass；可选接口以扩展 Protocol 表达。
- 光标列和字符串长度保留 TS 的 UTF-16 单位，通过 `_javascript.py` 处理 astral 字符和孤立 surrogate。
- 文件补全保留 `fd` 查询参数、两轮搜索、取消、排序、引用与路径前缀处理；不安装或替代用户指定的 `fd`。
- 分词、Unicode 属性和 locale 排序使用 `PyICU==2.16.2`，系统 ICU 至少为 70。
  与 Node 的 ICU/Unicode 数据版本及默认 locale 相同是严格一致的前提。
  宽字符范围直接移植 `get-east-asian-width@1.6.0`，不依赖系统 Unicode 版本。
- 撤销栈用 `deepcopy` 复制 Python 状态，保留内部引用关系及循环；Python 自定义对象的复制协议不等同于 JS structuredClone。
- `AbortSignal` 通过结构化协议接入，无需依赖 AI/Chord；调用者可传入这些包的信号。异步执行使用 asyncio。
- 原生剪贴板直接通过 ctypes 调用 AppKit/CoreGraphics、Windows API 或 libxcb；保留不可用 `UNDEFINED`
  与无内容 `None` 的区别，以及 X11 的传输期限、私有线程等待上限和忙状态。无需 Node/N-API。
- TUI 基类保留覆盖层焦点恢复、鼠标分发、输入过滤、渲染节流、立即输入帧、颜色查询和光标标记。
  debug 的源默认键放入 `DEFAULT_APP_KEYBINDINGS`，构造时可覆盖。
- Editor 源码中的四个直接按键分支收进 `DEFAULT_EDITOR_KEYBINDINGS` 并并入注册表，
  默认仍为 Shift+Backspace、Shift+Delete、Shift+Space 和 Enter；这是按仓库要求保持键位可配置的适配。
- `ProcessTerminal` 默认接真实 POSIX/Windows 终端，也可注入 `ProcessIO`；保留协议协商、
  分片等待、输入排空、原始模式、尺寸事件、调试写日志和 OSC 进度心跳。
  显式的 COLUMNS/LINES 环境值按源保留浮点数和非有限值，不在终端契约层截断。
- 主屏渲染保留逐行差分、Kitty 图像删除、光标定位及可恢复状态；写出按 UTF-16 单元分块并保护 surrogate pair。
- Markdown 使用原生 Marked 18.0.11 Lexer/Tokenizer，包含 GFM 表格、任务列表、HTML token、
  引用链接、嵌套强调和扩展 tokenizer；Token 使用可扩展 dataclass。扩展的 context 参数替代 JS this，
  start 的位置仍按 UTF-16 计算。私有 regex 适配器仅覆盖固定 grammar 实际用到的 JS 正则语法。
- Editor 保留多行编辑、原子粘贴标记、历史草稿、撤销/kill/yank、鼠标和串行异步补全。
  无活动 asyncio loop 时使用后台 daemon loop；单个不可拆字素宽于行宽时的源递归边界仍保留。
- Marked 公开入口包含 HTML Parser/Renderer/TextRenderer/Hooks、use 覆盖链、walk_tokens 与异步解析；
  async 选项用 `async_` 表示，扩展方法显式接收 context。在活动 loop 中异步解析立即调度 Task，
  否则返回 awaitable，由调用者运行；词法扫描使用同一套原生 Python 实现。
- 全屏渲染包含搜索、选区、捕获、滚动条、自动滚动、剪贴板、Kitty 缓存淘汰和退出后的文档恢复。
  异步剪贴板优先在活动 loop 中执行，否则通过后台线程承接 awaitable。
- 定时器优先使用活动 asyncio loop；同步调用时使用 Python 线程计时器。线程调度、操作系统 ABI、
  原生剪贴板错误和非标准数值输入均等待统一验证。
- 工作区记录 PyICU sdist 的实际空运行时依赖 metadata，使 `uv lock` 无需运行其构建；本轮未安装或构建 PyICU。

完整实现结束后统一执行差异测试，重点覆盖 UTF-16、ANSI、Unicode 分词、终端协议与平台集成。
