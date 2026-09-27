"""Declarative Marked 18.0.11 regex grammar, translated from src/rules.ts.

See _marked_regex.py for the source license and ECMAScript matching adapter.
"""

from ._marked_regex import Regex as _Regex, Edit as _Edit, NOOP as _NOOP

def _create_paragraph(list_interrupt: _Regex) -> _Regex:
    return (_Edit(_paragraph)
        .replace('hr', hr)
        .replace('heading', r' {0,3}#{1,6}(?:\s|$)')
        .replace('|lheading', '').replace('|table', '')
        .replace('blockquote', ' {0,3}>')
        .replace('fences', r' {0,3}(?:`{3,}(?=[^`\n]*(?:\n|$))|~~~)[^\n]*(?:\n|$)')
        .replace('list', list_interrupt)
        .replace('html', r'</?(?:tag)(?: +|\n|/?>)|<(?:script|pre|style|textarea|!--)')
        .replace('tag', _tag).get_regex())

newline = _Regex('^(?:[ \\t]*(?:\\n|$))+', '')

blockCode = _Regex('^((?: {4}| {0,3}\\t)[^\\n]+(?:\\n(?:[ \\t]*(?:\\n|$))*)?)+', '')

fences = _Regex('^ {0,3}(`{3,}(?=[^`\\n]*(?:\\n|$))|~{3,})([^\\n]*)(?:\\n|$)(?:|([\\s\\S]*?)(?:\\n|$))(?: {0,3}\\1[~`]* *(?=\\n|$)|$)', '')

hr = _Regex('^ {0,3}((?:-[\\t ]*){3,}|(?:_[ \\t]*){3,}|(?:\\*[ \\t]*){3,})(?:\\n+|$)', '')

heading = _Regex('^ {0,3}(#{1,6})(?=\\s|$)(.*)(?:\\n+|$)', '')

bullet = _Regex(' {0,3}(?:[*+-]|\\d{1,9}[.)])', '')

lheadingCore = _Regex('^(?!bull |blockCode|fences|blockquote|heading|html|table)((?:.|\\n(?!\\s*?\\n|bull |blockCode|fences|blockquote|heading|html|table))+?)\\n {0,3}(=+|-+) *(?:\\n+|$)', '')

lheading = _Edit ( lheadingCore ) . replace ( _Regex('bull', 'g') , bullet ) . replace ( _Regex('blockCode', 'g') , _Regex('(?: {4}| {0,3}\\t)', '') ) . replace ( _Regex('fences', 'g') , _Regex(' {0,3}(?:`{3,}|~{3,})', '') ) . replace ( _Regex('blockquote', 'g') , _Regex(' {0,3}>', '') ) . replace ( _Regex('heading', 'g') , _Regex(' {0,3}#{1,6}(?:\\s|$)', '') ) . replace ( _Regex('html', 'g') , _Regex(' {0,3}<[^\\n>]+>\\n', '') ) . replace ( _Regex('\\|table', 'g') , '' ) . get_regex ( )

lheadingGfm = _Edit ( lheadingCore ) . replace ( _Regex('bull', 'g') , bullet ) . replace ( _Regex('blockCode', 'g') , _Regex('(?: {4}| {0,3}\\t)', '') ) . replace ( _Regex('fences', 'g') , _Regex(' {0,3}(?:`{3,}|~{3,})', '') ) . replace ( _Regex('blockquote', 'g') , _Regex(' {0,3}>', '') ) . replace ( _Regex('heading', 'g') , _Regex(' {0,3}#{1,6}(?:\\s|$)', '') ) . replace ( _Regex('html', 'g') , _Regex(' {0,3}<[^\\n>]+>\\n', '') ) . replace ( _Regex('table', 'g') , _Regex(' {0,3}\\|?(?:[:\\- ]*\\|)+[\\:\\- ]*\\n', '') ) . get_regex ( )

_paragraph = _Regex('^([^\\n]+(?:\\n(?!hr|heading|lheading|blockquote|fences|list|html|table|[ \\t]+\\n)[^\\n]+)*)', '')

blockText = _Regex('^[^\\n]+', '')

_blockLabel = _Regex('(?!\\s*\\])(?:\\\\[\\s\\S]|[^\\[\\]\\\\])+', '')

_definition = _Edit ( _Regex('^ {0,3}\\[(label)\\]: *(?:\\n[ \\t]*)?([^<\\s][^\\s]*|<.*?>)(?:(?: +(?:\\n[ \\t]*)?| *\\n[ \\t]*)(title))? *(?:\\n+|$)', '') ) . replace ( 'label' , _blockLabel ) . replace ( 'title' , _Regex('(?:"(?:\\\\"?|[^"\\\\])*"|\'[^\'\\n]*(?:\\n[^\'\\n]+)*\\n?\'|\\([^()]*\\))', '') ) . get_regex ( )

list = _Edit ( _Regex('^(bull)([ \\t][^\\n]*?)?(?:\\n|$)', '') ) . replace ( _Regex('bull', 'g') , bullet ) . get_regex ( )

_tag = 'address|article|aside|base|basefont|blockquote|body|caption' + '|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption' + '|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe' + '|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option' + '|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title' + '|tr|track|ul'

_comment = _Regex('<!--(?:-?>|[\\s\\S]*?(?:-->|$))', '')

html = _Edit ( '^ {0,3}(?:' + '<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n*|$)' + '|comment[^\\n]*(\\n+|$)' + '|<\\?[\\s\\S]*?(?:\\?>[^\\n]*\\n*|$)' + '|<![A-Z][\\s\\S]*?(?:>[^\\n]*\\n*|$)' + '|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>[^\\n]*\\n*|$)' + '|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$)' + '|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$)' + '|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$)' + ')' , 'i' ) . replace ( 'comment' , _comment ) . replace ( 'tag' , _tag ) . replace ( 'attribute' , _Regex(' +[a-zA-Z:_][\\w.:-]*(?: *= *"[^"\\n]*"| *= *\'[^\'\\n]*\'| *= *[^\\s"\'=<>`]+)?', '') ) . get_regex ( )

paragraph = _create_paragraph ( _Regex(' {0,3}(?:[*+-]|1[.)])[ \\t]+[^ \\t\\n]', '') )

blockquoteParagraph = _create_paragraph ( _Regex(' {0,3}(?:[*+-]|\\d{1,9}[.)])(?:[ \\t]|\\n|$)', '') )

blockquote = _Edit ( _Regex('^( {0,3}> ?(paragraph|[^\\n]*)(?:\\n|$))+', '') ) . replace ( 'paragraph' , blockquoteParagraph ) . get_regex ( )

blockNormal = {
    'blockquote': blockquote,
    'code': blockCode,
    'def': _definition,
    'fences': fences,
    'heading': heading,
    'hr': hr,
    'html': html,
    'lheading': lheading,
    'list': list,
    'newline': newline,
    'paragraph': paragraph,
    'table': _NOOP,
    'text': blockText
}

gfmTable = _Edit ( '^ *([^\\n ].*)\\n' + ' {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)' + '(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)' ) . replace ( 'hr' , hr ) . replace ( 'heading' , ' {0,3}#{1,6}(?:\\s|$)' ) . replace ( 'blockquote' , ' {0,3}>' ) . replace ( 'code' , '(?: {4}| {0,3}\t)[^\\n]' ) . replace ( 'fences' , ' {0,3}(?:`{3,}(?=[^`\\n]*(?:\\n|$))|~~~)[^\\n]*(?:\\n|$)' ) . replace ( 'list' , ' {0,3}(?:[*+-]|1[.)])[ \\t]' ) . replace ( 'html' , '</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)' ) . replace ( 'tag' , _tag ) . get_regex ( )

blockGfm = {
    **blockNormal,
    'lheading': lheadingGfm,
    'table': gfmTable,
    'paragraph': _Edit ( _paragraph ) . replace ( 'hr' , hr ) . replace ( 'heading' , ' {0,3}#{1,6}(?:\\s|$)' ) . replace ( '|lheading' , '' ) . replace ( 'table' , gfmTable ) . replace ( 'blockquote' , ' {0,3}>' ) . replace ( 'fences' , ' {0,3}(?:`{3,}(?=[^`\\n]*(?:\\n|$))|~~~)[^\\n]*(?:\\n|$)' ) . replace ( 'list' , ' {0,3}(?:[*+-]|1[.)])[ \\t]+[^ \\t\\n]' ) . replace ( 'html' , '</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)' ) . replace ( 'tag' , _tag ) . get_regex ( )
}

blockPedantic = {
    **blockNormal,
    'html': _Edit ( '^ *(?:comment *(?:\\n|\\s*$)' + '|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)' + '|<tag(?:"[^"]*"|\'[^\']*\'|\\s[^\'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))' ) . replace ( 'comment' , _comment ) . replace ( _Regex('tag', 'g') , '(?!(?:' + 'a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub' + '|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)' + '\\b)\\w+(?!:|[^\\w\\s@]*@)\\b' ) . get_regex ( ),
    'def': _Regex('^ *\\[([^\\]]+)\\]: *<?([^\\s>]+)>?(?: +(["(][^\\n]+[")]))? *(?:\\n+|$)', ''),
    'heading': _Regex('^(#{1,6})(.*)(?:\\n+|$)', ''),
    'fences': _NOOP,
    'lheading': _Regex('^(.+?)\\n {0,3}(=+|-+) *(?:\\n+|$)', ''),
    'paragraph': _Edit ( _paragraph ) . replace ( 'hr' , hr ) . replace ( 'heading' , ' *#{1,6} *[^\n]' ) . replace ( 'lheading' , lheading ) . replace ( '|table' , '' ) . replace ( 'blockquote' , ' {0,3}>' ) . replace ( '|fences' , '' ) . replace ( '|list' , '' ) . replace ( '|html' , '' ) . replace ( '|tag' , '' ) . get_regex ( )
}

escape = _Regex('^\\\\([!"#$%&\'()*+,\\-./:;<=>?@\\[\\]\\\\^_`{|}~])', '')

inlineCode = _Regex('^(`+)([^`]|[^`][\\s\\S]*?[^`])\\1(?!`)', '')

br = _Regex('^( {2,}|\\\\)\\n(?!\\s*$)', '')

inlineText = _Regex('^(`+|[^`])(?:(?= {2,}\\n)|[\\s\\S]*?(?:(?=[\\\\<!\\[`*_]|\\b_|$)|[^ ](?= {2,}\\n)))', '')

_punctuation = _Regex('[\\p{P}\\p{S}]', 'u')

_punctuationOrSpace = _Regex('[\\s\\p{P}\\p{S}]', 'u')

_notPunctuationOrSpace = _Regex('[^\\s\\p{P}\\p{S}]', 'u')

punctuation = _Edit ( _Regex('^((?![*_])punctSpace)', '') , 'u' ) . replace ( _Regex('punctSpace', 'g') , _punctuationOrSpace ) . get_regex ( )

_openQuote = _Regex('[\\p{Pi}\\p{Ps}"\']', 'u')

_punctuationGfmStrongEm = _Regex('(?!~)[\\p{P}\\p{S}]', 'u')

_punctuationOrSpaceGfmStrongEm = _Regex('(?!~)[\\s\\p{P}\\p{S}]', 'u')

_notPunctuationOrSpaceGfmStrongEm = _Regex('(?:[^\\s\\p{P}\\p{S}]|~)', 'u')

blockSkip = _Edit ( _Regex('link|precode-code|html', '') , 'g' ) . replace ( 'link' , _Regex('\\[(?:[^\\[\\]`]|(?<a>`+)[^`]+\\k<a>(?!`))*?\\]\\((?:\\\\[\\s\\S]|[^\\\\\\(\\)]|\\((?:\\\\[\\s\\S]|[^\\\\\\(\\)])*\\))*\\)', '') ) . replace ( 'precode-' , '(?<!`)()' ) . replace ( 'code' , _Regex('(?<b>`+)[^`]+\\k<b>(?!`)', '') ) . replace ( 'html' , _Regex('<(?! )[^<>]*?>', '') ) . get_regex ( )

emStrongLDelimCore = _Regex('^(?:\\*+(?:((?!\\*)punct)|([^\\s*]))?)|^_+(?:((?!_)punct)|([^\\s_]))?', '')

emStrongLDelim = _Edit ( emStrongLDelimCore , 'u' ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

emStrongLDelimGfm = _Edit ( emStrongLDelimCore , 'u' ) . replace ( _Regex('punct', 'g') , _punctuationGfmStrongEm ) . get_regex ( )

emStrongLDelimPedanticCore = _Regex('^(?:\\*+(?:((?!\\*)(?!openQuote)punct)|([^\\s*]))?)|^_+(?:((?!_)(?!openQuote)punct)|([^\\s_]))?', '')

emStrongLDelimPedantic = _Edit ( emStrongLDelimPedanticCore , 'u' ) . replace ( _Regex('openQuote', 'g') , _openQuote ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

emStrongRDelimAstCore = '^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)' + '|[^*]+(?=[^*])' + '|(?!\\*)punct(\\*+)(?=[\\s]|$)' + '|notPunctSpace(\\*+)(?!\\*)(?=punctSpace|$)' + '|(?!\\*)punctSpace(\\*+)(?=notPunctSpace)' + '|[\\s](\\*+)(?!\\*)(?=punct)' + '|(?!\\*)punct(\\*+)(?!\\*)(?=punct)' + '|notPunctSpace(\\*+)(?=notPunctSpace)'

emStrongRDelimAst = _Edit ( emStrongRDelimAstCore , 'gu' ) . replace ( _Regex('notPunctSpace', 'g') , _notPunctuationOrSpace ) . replace ( _Regex('punctSpace', 'g') , _punctuationOrSpace ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

emStrongRDelimAstGfm = _Edit ( emStrongRDelimAstCore , 'gu' ) . replace ( _Regex('notPunctSpace', 'g') , _notPunctuationOrSpaceGfmStrongEm ) . replace ( _Regex('punctSpace', 'g') , _punctuationOrSpaceGfmStrongEm ) . replace ( _Regex('punct', 'g') , _punctuationGfmStrongEm ) . get_regex ( )

emStrongRDelimAstPedanticCore = '^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)' + '|[^*]+(?=[^*])' + '|(?!\\*)punct(\\*+)(?=[\\s]|$)' + '|notPunctSpace(\\*+)(?!\\*)(?=punctSpace|$)' + '|(?!\\*)[\\s](\\*+)(?=notPunctSpace)' + '|[\\s](\\*+)(?!\\*)(?=punct)' + '|(?!\\*)punct(\\*+)(?!\\*)(?=punct)' + '|(?:(?!\\*)punct|notPunctSpace)(\\*+)(?!\\*)(?=notPunctSpace)'

emStrongRDelimAstPedantic = _Edit ( emStrongRDelimAstPedanticCore , 'gu' ) . replace ( _Regex('notPunctSpace', 'g') , _notPunctuationOrSpace ) . replace ( _Regex('punctSpace', 'g') , _punctuationOrSpace ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

emStrongRDelimUnd = _Edit ( '^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)' + '|[^_]+(?=[^_])' + '|(?!_)punct(_+)(?=[\\s]|$)' + '|notPunctSpace(_+)(?!_)(?=punctSpace|$)' + '|(?!_)punctSpace(_+)(?=notPunctSpace)' + '|[\\s](_+)(?!_)(?=punct)' + '|(?!_)punct(_+)(?!_)(?=punct)' , 'gu' ) . replace ( _Regex('notPunctSpace', 'g') , _notPunctuationOrSpace ) . replace ( _Regex('punctSpace', 'g') , _punctuationOrSpace ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

emStrongRDelimUndPedanticCore = '^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)' + '|[^_]+(?=[^_])' + '|(?!_)punct(_+)(?=[\\s]|$)' + '|notPunctSpace(_+)(?!_)(?=punctSpace|$)' + '|(?!_)[\\s](_+)(?=notPunctSpace)' + '|[\\s](_+)(?!_)(?=punct)' + '|(?!_)punct(_+)(?!_)(?=punct)' + '|(?:(?!_)punct|notPunctSpace)(_+)(?!_)(?=notPunctSpace)'

emStrongRDelimUndPedantic = _Edit ( emStrongRDelimUndPedanticCore , 'gu' ) . replace ( _Regex('notPunctSpace', 'g') , _notPunctuationOrSpace ) . replace ( _Regex('punctSpace', 'g') , _punctuationOrSpace ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

delLDelim = _Edit ( _Regex('^~~?(?:((?!~)punct)|[^\\s~])', '') , 'u' ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

delRDelimCore = '^[^~]+(?=[^~])' + '|(?!~)punct(~~?)(?=[\\s]|$)' + '|notPunctSpace(~~?)(?!~)(?=punctSpace|$)' + '|(?!~)punctSpace(~~?)(?=notPunctSpace)' + '|[\\s](~~?)(?!~)(?=punct)' + '|(?!~)punct(~~?)(?!~)(?=punct)' + '|notPunctSpace(~~?)(?=notPunctSpace)'

delRDelim = _Edit ( delRDelimCore , 'gu' ) . replace ( _Regex('notPunctSpace', 'g') , _notPunctuationOrSpace ) . replace ( _Regex('punctSpace', 'g') , _punctuationOrSpace ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

anyPunctuation = _Edit ( _Regex('\\\\(punct)', '') , 'gu' ) . replace ( _Regex('punct', 'g') , _punctuation ) . get_regex ( )

autolink = _Edit ( _Regex('^<(scheme:[^\\s\\x00-\\x1f<>]*|email)>', '') ) . replace ( 'scheme' , _Regex('[a-zA-Z][a-zA-Z0-9+.-]{1,31}', '') ) . replace ( 'email' , _Regex("[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])", '') ) . get_regex ( )

_inlineComment = _Edit ( _comment ) . replace ( '(?:-->|$)' , '-->' ) . get_regex ( )

tag = _Edit ( '^comment' + '|^</[a-zA-Z][\\w:-]*\\s*>' + '|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>' + '|^<\\?[\\s\\S]*?\\?>' + '|^<![a-zA-Z]+\\s[\\s\\S]*?>' + '|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>' ) . replace ( 'comment' , _inlineComment ) . replace ( 'attribute' , _Regex('\\s+[a-zA-Z:_][\\w.:-]*(?:\\s*=\\s*"[^"]*"|\\s*=\\s*\'[^\']*\'|\\s*=\\s*[^\\s"\'=<>`]+)?', '') ) . get_regex ( )

_inlineLabel = _Regex('(?:\\[(?:\\\\[\\s\\S]|[^\\[\\]\\\\])*\\]|\\\\[\\s\\S]|`+(?!`)[^`]*?`+(?!`)|``+(?=\\])|[^\\[\\]\\\\`])*?', '')

link = _Edit ( _Regex('^!?\\[(label)\\]\\(\\s*(href)(?:(?:[ \\t]+(?:\\n[ \\t]*)?|\\n[ \\t]*)(title))?\\s*\\)', '') ) . replace ( 'label' , _inlineLabel ) . replace ( 'href' , _Regex('<(?:\\\\.|[^\\n<>\\\\])+>|[^ \\t\\n\\x00-\\x1f]+|(?=\\))', '') ) . replace ( 'title' , _Regex('"(?:\\\\"?|[^"\\\\])*"|\'(?:\\\\\'?|[^\'\\\\])*\'|\\((?:\\\\\\)?|[^)\\\\])*\\)', '') ) . get_regex ( )

reflink = _Edit ( _Regex('^!?\\[(label)\\]\\[(ref)\\]', '') ) . replace ( 'label' , _inlineLabel ) . replace ( 'ref' , _blockLabel ) . get_regex ( )

nolink = _Edit ( _Regex('^!?\\[(ref)\\](?:\\[\\])?', '') ) . replace ( 'ref' , _blockLabel ) . get_regex ( )

reflinkSearch = _Edit ( 'reflink|nolink(?!\\()' , 'g' ) . replace ( 'reflink' , reflink ) . replace ( 'nolink' , nolink ) . get_regex ( )

_caseInsensitiveProtocol = _Regex('[hH][tT][tT][pP][sS]?|[fF][tT][pP]', '')

inlineNormal = {
    '_backpedal': _NOOP,
    'anyPunctuation': anyPunctuation,
    'autolink': autolink,
    'blockSkip': blockSkip,
    'br': br,
    'code': inlineCode,
    'del': _NOOP,
    'delLDelim': _NOOP,
    'delRDelim': _NOOP,
    'emStrongLDelim': emStrongLDelim,
    'emStrongRDelimAst': emStrongRDelimAst,
    'emStrongRDelimUnd': emStrongRDelimUnd,
    'escape': escape,
    'link': link,
    'nolink': nolink,
    'punctuation': punctuation,
    'reflink': reflink,
    'reflinkSearch': reflinkSearch,
    'tag': tag,
    'text': inlineText,
    'url': _NOOP
}

inlinePedantic = {
    **inlineNormal,
    'emStrongLDelim': emStrongLDelimPedantic,
    'emStrongRDelimAst': emStrongRDelimAstPedantic,
    'emStrongRDelimUnd': emStrongRDelimUndPedantic,
    'link': _Edit ( _Regex('^!?\\[(label)\\]\\((.*?)\\)', '') ) . replace ( 'label' , _inlineLabel ) . get_regex ( ),
    'reflink': _Edit ( _Regex('^!?\\[(label)\\]\\s*\\[([^\\]]*)\\]', '') ) . replace ( 'label' , _inlineLabel ) . get_regex ( )
}

inlineGfm = {
    **inlineNormal,
    'emStrongRDelimAst': emStrongRDelimAstGfm,
    'emStrongLDelim': emStrongLDelimGfm,
    'delLDelim': delLDelim,
    'delRDelim': delRDelim,
    'url': _Edit ( _Regex('^((?:protocol):\\/\\/|www\\.)(?:[a-zA-Z0-9\\-]+\\.?)+[^\\s<]*|^email', '') ) . replace ( 'protocol' , _caseInsensitiveProtocol ) . replace ( 'email' , _Regex('[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])', '') ) . get_regex ( ),
    '_backpedal': _Regex('(?:[^?!.,:;*_\'"~()&]+|\\([^)]*\\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_\'"~)]+(?!$))+', ''),
    'del': _Regex('^(~~?)(?=[^\\s~])((?:\\\\[\\s\\S]|[^\\\\])*?(?:\\\\[\\s\\S]|[^\\s~\\\\]))\\1(?=[^~]|$)', ''),
    'text': _Edit ( _Regex("^(`+|~+|[^`~])(?:(?=[`~])|(?= {2,}\\n)|(?=[a-zA-Z0-9.!#$%&'*+\\/=?_`{\\|}~-]+@)|[\\s\\S]*?(?:(?=[\\\\<!\\[`*~_]|\\b_|protocol:\\/\\/|www\\.|$)|[^ ](?= {2,}\\n)|[^a-zA-Z0-9.!#$%&'*+\\/=?_`{\\|}~-](?=[a-zA-Z0-9.!#$%&'*+\\/=?_`{\\|}~-]+@)))", '') ) . replace ( 'protocol' , _caseInsensitiveProtocol ) . get_regex ( )
}

inlineBreaks = {
    **inlineGfm,
    'br': _Edit ( br ) . replace ( '{2,}' , '*' ) . get_regex ( ),
    'text': _Edit ( inlineGfm['text'] ) . replace ( '\\b_' , '\\b_| {2,}\\n' ) . replace ( _Regex('\\{2,\\}', 'g') , '*' ) . get_regex ( )
}

block = {
    'normal': blockNormal,
    'gfm': blockGfm,
    'pedantic': blockPedantic
}

inline = {
    'normal': inlineNormal,
    'gfm': inlineGfm,
    'breaks': inlineBreaks,
    'pedantic': inlinePedantic
}

other = {
    'codeRemoveIndent': _Regex('^(?: {1,4}| {0,3}\\t)', 'gm'),
    'outputLinkReplace': _Regex('\\\\([\\[\\]])', 'g'),
    'indentCodeCompensation': _Regex('^(\\s+)(?:```)', ''),
    'beginningSpace': _Regex('^\\s+', ''),
    'endingHash': _Regex('#$', ''),
    'startingSpaceChar': _Regex('^ ', ''),
    'endingSpaceChar': _Regex(' $', ''),
    'nonSpaceChar': _Regex('[^ ]', ''),
    'newLineCharGlobal': _Regex('\\n', 'g'),
    'tabCharGlobal': _Regex('\\t', 'g'),
    'multipleSpaceGlobal': _Regex('\\s+', 'g'),
    'blankLine': _Regex('^[ \\t]*$', ''),
    'doubleBlankLine': _Regex('\\n[ \\t]*\\n[ \\t]*$', ''),
    'blockquoteStart': _Regex('^ {0,3}>', ''),
    'blockquoteSetextReplace': _Regex('\\n {0,3}((?:=+|-+) *)(?=\\n|$)', 'g'),
    'blockquoteSetextReplace2': _Regex('^ {0,3}>[ \\t]?', 'gm'),
    'listReplaceNesting': _Regex('^ {1,4}(?=( {4})*[^ ])', 'g'),
    'listIsTask': _Regex('^\\[[ xX]\\] +\\S', ''),
    'listReplaceTask': _Regex('^\\[[ xX]\\] +', ''),
    'listTaskCheckbox': _Regex('\\[[ xX]\\]', ''),
    'anyLine': _Regex('\\n.*\\n', ''),
    'hrefBrackets': _Regex('^<(.*)>$', ''),
    'tableDelimiter': _Regex('[:|]', ''),
    'tableAlignChars': _Regex('^\\||\\| *$', 'g'),
    'tableRowBlankLine': _Regex('\\n[ \\t]*$', ''),
    'tableAlignRight': _Regex('^ *-+: *$', ''),
    'tableAlignCenter': _Regex('^ *:-+: *$', ''),
    'tableAlignLeft': _Regex('^ *:-+ *$', ''),
    'startATag': _Regex('^<a ', 'i'),
    'endATag': _Regex('^<\\/a>', 'i'),
    'startPreScriptTag': _Regex('^<(pre|code|kbd|script)(\\s|>)', 'i'),
    'endPreScriptTag': _Regex('^<\\/(pre|code|kbd|script)(\\s|>)', 'i'),
    'startAngleBracket': _Regex('^<', ''),
    'endAngleBracket': _Regex('>$', ''),
    'pedanticHrefTitle': _Regex('^([^\'"]*[^\\s])\\s+([\'"])(.*)\\2', ''),
    'unicodeAlphaNumeric': _Regex('[\\p{L}\\p{N}]', 'u'),
    'escapeTest': _Regex('[&<>"\']', ''),
    'escapeReplace': _Regex('[&<>"\']', 'g'),
    'escapeTestNoEncode': _Regex('[<>"\']|&(?!(#\\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\\w+);)', ''),
    'escapeReplaceNoEncode': _Regex('[<>"\']|&(?!(#\\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\\w+);)', 'g'),
    'caret': _Regex('(^|[^\\[])\\^', 'g'),
    'percentDecode': _Regex('%25', 'g'),
    'findPipe': _Regex('\\|', 'g'),
    'splitPipe': _Regex(' \\|', ''),
    'slashPipe': _Regex('\\\\\\|', 'g'),
    'carriageReturn': _Regex('\\r\\n|\\r', 'g'),
    'spaceLine': _Regex('^ +$', 'gm'),
    'notSpaceStart': _Regex('^\\S*', ''),
    'endingNewline': _Regex('\\n$', ''),
}
