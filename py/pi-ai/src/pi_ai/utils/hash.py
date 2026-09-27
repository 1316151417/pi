"""Fast deterministic string hash ported from ``src/utils/hash.ts``."""

__all__ = ["short_hash"]


def short_hash(text: str) -> str:
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    for offset in range(0, len(encoded), 2):
        character = encoded[offset] | encoded[offset + 1] << 8
        h1 = ((h1 ^ character) * 2654435761) & 0xFFFFFFFF
        h2 = ((h2 ^ character) * 1597334677) & 0xFFFFFFFF
    h1 = (((h1 ^ (h1 >> 16)) * 2246822507) ^ ((h2 ^ (h2 >> 13)) * 3266489909)) & 0xFFFFFFFF
    h2 = (((h2 ^ (h2 >> 16)) * 2246822507) ^ ((h1 ^ (h1 >> 13)) * 3266489909)) & 0xFFFFFFFF
    parts: list[str] = []
    for value in (h2, h1):
        digits = ""
        while value:
            value, remainder = divmod(value, 36)
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"[remainder] + digits
        parts.append(digits or "0")
    return "".join(parts)
