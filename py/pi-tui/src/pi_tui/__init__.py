"""Public APIs ported from ``@earendil-works/pi-tui``.

The implementation is being filled in dependency order; see the package README.
"""

from .abort import AbortController, AbortError, AbortSignal
from ._markdown_lexer import (
    Alignment, Hooks, HooksObject, Lexer, LinkDefinition, Links, Marked, MarkedExtension, MarkedExtensions,
    MarkedOptions, MarkedToken, MaybePromise, Parser, Renderer, RendererContext, RendererExtension,
    RendererExtensionFunction, RendererObject, RendererThis, TableCell, TableRow, TextRenderer, Token,
    Tokenizer, TokenizerAndRendererExtension, TokenizerContext, TokenizerExtension, TokenizerExtensionFunction,
    TokenizerObject, TokenizerStartFunction, TokenizerThis, Tokens, TokensList, WalkTokensCallback,
)
from .alt_screen_search import (
    AltScreenSearchComponent, AltScreenSearchIndex, AltScreenSearchMatch, AltScreenSearchResult,
    AltScreenSearchSegment, find_alt_screen_search_matches, get_alt_screen_search_match_key,
)
from .autocomplete import (
    AutocompleteAbortSignal,
    AutocompleteItem,
    AutocompleteOptions,
    AutocompleteProvider,
    AutocompleteSuggestions,
    CombinedAutocompleteProvider,
    CompletionResult,
    FileAutocompleteProvider,
    SlashCommand,
    TriggeredAutocompleteProvider,
)
from .components import (
    AltScreenFlashContainer, Box, CancellableLoader, HStack, Image, ImageOptions, ImageTheme,
    Editor, EditorCursor, EditorOptions, EditorTheme, TextChunk, word_wrap_line,
    Input, InputOptions, Loader, LoaderIndicatorOptions, DefaultTextStyle, Markdown, MarkdownOptions,
    MarkdownTheme, MouseRegion, MouseRegionHandler,
    ScrollView, ScrollViewOptions, ScrollViewScrollbar, ScrollViewScrollToOptions,
    SelectItem, SelectList, SelectListLayoutOptions, SelectListTheme, SelectListTruncatePrimaryContext,
    SettingItem, SettingsList, SettingsListOptions, SettingsListTheme, SettingSubmenuCloseOptions,
    SettingSubmenuDone, Spacer, Stack, StackChild, StackEntry, StackEntryOptions, StackOptions,
    Text, TruncatedText, VStack,
)
from .editor_component import (
    AutocompleteEditorComponent, EditorComponent, ExpandedTextEditorComponent, HistoryEditorComponent,
    InsertionEditorComponent, StyledEditorComponent,
)
from .fuzzy import FuzzyMatch, fuzzy_filter, fuzzy_match
from .keybindings import (
    DEFAULT_EDITOR_KEYBINDINGS, TUI_KEYBINDINGS, Keybinding, KeybindingConflict, KeybindingDefinition, KeybindingDefinitions,
    Keybindings, KeybindingsConfig, KeybindingsManager, get_keybindings, set_keybindings,
)
from .keys import (
    Key, KeyEventType, KeyId, decode_kitty_printable, decode_printable_key, is_key_release, is_key_repeat,
    is_kitty_protocol_active, matches_key, parse_key, set_kitty_protocol_active,
)
from .kill_ring import KillRing, KillRingPushOptions
from .latex import RenderLatexOptions, render_latex
from .layout import (
    LayoutBox, LayoutFrame, LayoutRect, ScrollbarGeometry, get_layout_boxes_at,
    get_scroll_view_box, get_scroll_views_at, get_scrollbar_geometry, render_layout_frame,
)
from .native_platform import UNDEFINED, ModifierKey, NativeClipboard, NativePlatformHelper, Undefined, get_native_clipboard, get_native_platform_helper
from .stdin_buffer import StdinBuffer, StdinBufferEventMap, StdinBufferOptions
from .terminal import ProcessTerminal, Terminal
from .terminal_colors import RgbColor, TerminalColorScheme, parse_osc11_background_color, parse_terminal_color_scheme_report
from .terminal_image import (
    CapabilityOverrides, CellDimensions, ImageDimensions, ImageProtocol, ImageRenderOptions,
    TerminalCapabilities, allocate_image_id, calculate_image_rows, delete_all_kitty_images,
    delete_kitty_image, detect_capabilities, encode_iterm2, encode_kitty, get_capabilities,
    get_cell_dimensions, get_gif_dimensions, get_image_dimensions, get_jpeg_dimensions,
    get_png_dimensions, get_webp_dimensions, hyperlink, image_fallback, render_image,
    reset_capabilities_cache, set_capabilities, set_capability_overrides, set_cell_dimensions,
)
from .tui import (
    CURSOR_MARKER, DEFAULT_APP_KEYBINDINGS, VIEWPORT_TUI, AppKeybindings, Component, Container,
    Focusable, OverlayAnchor, OverlayBounds, OverlayHandle, OverlayMargin, OverlayOptions,
    OverlayUnfocusOptions, SizeValue, TUI, TuiBase, TuiInputListener, TuiInputListenerResult,
    TuiMode, TuiMouseButton, TuiMouseDispatchResult, TuiMouseDispatchTarget, TuiMouseEvent,
    TuiMouseEventResult, TuiMouseEventType, TuiStopOptions, ViewportTUI, composite_tui_line,
    dispatch_mouse_event, is_focusable, is_viewport_tui, retarget_mouse_event,
)
from .undo_stack import UndoStack
from .tui_main_screen import TuiMainScreen, TuiMainScreenRenderState
from .tui_alt_screen import TuiAltScreen, TuiAltScreenOptions
from .utils import get_osc8_link_at_column, slice_by_column, strip_terminal_sequences, truncate_to_width, visible_width, wrap_text_with_ansi

__all__ = [
    "Hooks", "HooksObject", "Links", "MarkedExtension", "MarkedExtensions", "MarkedToken", "MaybePromise",
    "Parser", "Renderer", "RendererContext", "RendererExtension", "RendererExtensionFunction", "RendererObject",
    "RendererThis", "TableRow", "TextRenderer", "TokenizerAndRendererExtension", "TokenizerExtensionFunction",
    "TokenizerObject", "TokenizerStartFunction", "TokenizerThis", "WalkTokensCallback",
    "Editor", "EditorCursor", "EditorOptions", "EditorTheme", "TextChunk", "word_wrap_line",
    "TuiAltScreen", "TuiAltScreenOptions",
    "Alignment", "Lexer", "LinkDefinition", "Marked", "MarkedOptions", "TableCell", "Token", "Tokenizer",
    "TokenizerContext", "TokenizerExtension", "Tokens", "TokensList", "DefaultTextStyle", "Markdown",
    "MarkdownOptions", "MarkdownTheme",
    "DEFAULT_EDITOR_KEYBINDINGS", "HStack", "ProcessTerminal", "Terminal",
    "TuiMainScreen", "TuiMainScreenRenderState", "LayoutBox", "LayoutFrame", "LayoutRect",
    "ScrollbarGeometry", "get_layout_boxes_at", "get_scroll_view_box", "get_scroll_views_at",
    "get_scrollbar_geometry", "render_layout_frame",
    "AutocompleteAbortSignal", "AutocompleteItem", "AutocompleteOptions",
    "AutocompleteProvider", "AutocompleteSuggestions", "CombinedAutocompleteProvider",
    "CompletionResult", "FileAutocompleteProvider", "SlashCommand", "TriggeredAutocompleteProvider",
    "FuzzyMatch", "fuzzy_filter", "fuzzy_match", "KillRing", "KillRingPushOptions", "UndoStack",
    "AbortController", "AbortError", "AbortSignal", "AltScreenSearchComponent", "AltScreenSearchIndex",
    "AltScreenSearchMatch", "AltScreenSearchResult", "AltScreenSearchSegment", "find_alt_screen_search_matches",
    "get_alt_screen_search_match_key", "AltScreenFlashContainer", "Box", "CancellableLoader", "Image",
    "ImageOptions", "ImageTheme", "Input", "InputOptions", "Loader", "LoaderIndicatorOptions",
    "MouseRegion", "MouseRegionHandler", "ScrollView", "ScrollViewOptions", "ScrollViewScrollbar",
    "ScrollViewScrollToOptions", "SelectItem", "SelectList", "SelectListLayoutOptions", "SelectListTheme",
    "SelectListTruncatePrimaryContext", "SettingItem", "SettingsList", "SettingsListOptions", "SettingsListTheme",
    "SettingSubmenuCloseOptions", "SettingSubmenuDone", "Spacer", "Stack", "StackChild", "StackEntry",
    "StackEntryOptions", "StackOptions", "Text", "TruncatedText", "VStack", "AutocompleteEditorComponent",
    "EditorComponent", "ExpandedTextEditorComponent", "HistoryEditorComponent", "InsertionEditorComponent",
    "StyledEditorComponent", "TUI_KEYBINDINGS", "Keybinding", "KeybindingConflict", "KeybindingDefinition",
    "KeybindingDefinitions", "Keybindings", "KeybindingsConfig", "KeybindingsManager", "get_keybindings",
    "set_keybindings", "Key", "KeyEventType", "KeyId", "decode_kitty_printable", "decode_printable_key",
    "is_key_release", "is_key_repeat", "is_kitty_protocol_active", "matches_key", "parse_key", "set_kitty_protocol_active",
    "RenderLatexOptions", "render_latex", "UNDEFINED", "Undefined", "ModifierKey", "NativeClipboard",
    "NativePlatformHelper", "get_native_clipboard", "get_native_platform_helper", "StdinBuffer", "StdinBufferEventMap",
    "StdinBufferOptions", "RgbColor", "TerminalColorScheme", "parse_osc11_background_color", "parse_terminal_color_scheme_report",
    "CapabilityOverrides", "CellDimensions", "ImageDimensions", "ImageProtocol", "ImageRenderOptions",
    "TerminalCapabilities", "allocate_image_id", "calculate_image_rows", "delete_all_kitty_images",
    "delete_kitty_image", "detect_capabilities", "encode_iterm2", "encode_kitty", "get_capabilities",
    "get_cell_dimensions", "get_gif_dimensions", "get_image_dimensions", "get_jpeg_dimensions",
    "get_png_dimensions", "get_webp_dimensions", "hyperlink", "image_fallback", "render_image",
    "reset_capabilities_cache", "set_capabilities", "set_capability_overrides", "set_cell_dimensions",
    "CURSOR_MARKER", "DEFAULT_APP_KEYBINDINGS", "VIEWPORT_TUI", "AppKeybindings", "Component", "Container",
    "Focusable", "OverlayAnchor", "OverlayBounds", "OverlayHandle", "OverlayMargin", "OverlayOptions",
    "OverlayUnfocusOptions", "SizeValue", "TUI", "TuiBase", "TuiInputListener", "TuiInputListenerResult",
    "TuiMode", "TuiMouseButton", "TuiMouseDispatchResult", "TuiMouseDispatchTarget", "TuiMouseEvent",
    "TuiMouseEventResult", "TuiMouseEventType", "TuiStopOptions", "ViewportTUI", "composite_tui_line",
    "dispatch_mouse_event", "is_focusable", "is_viewport_tui", "retarget_mouse_event", "get_osc8_link_at_column",
    "slice_by_column", "strip_terminal_sequences", "truncate_to_width", "visible_width", "wrap_text_with_ansi",
]
