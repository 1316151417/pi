"""OpenAI cache-key length handling from ``api/openai-prompt-cache.ts``."""

OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    if key is None:
        return None
    # JavaScript Array.from counts valid surrogate pairs as one code point, but
    # retains unpaired surrogates. Preserve the caller's original representation.
    index = 0
    count = 0
    while index < len(key) and count < OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        if (0xD800 <= ord(key[index]) <= 0xDBFF and index + 1 < len(key)
                and 0xDC00 <= ord(key[index + 1]) <= 0xDFFF):
            index += 2
        else:
            index += 1
        count += 1
    return key if index == len(key) else key[:index]
