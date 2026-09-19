"""Image helpers ported from ``harness/tools/image.ts``."""

from __future__ import annotations

import base64

__all__ = ["detect_supported_image_mime_type", "encode_base64", "SUPPORTED_IMAGE_MIME_TYPES"]

SUPPORTED_IMAGE_MIME_TYPES = (
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
)

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_GIF_MAGIC = b"GIF87a"
_GIF89_MAGIC = b"GIF89a"
_BMP_MAGIC = b"BM"


def detect_supported_image_mime_type(data: bytes) -> "str | None":
    """Detect the image type by magic bytes; ``None`` when unsupported."""
    if len(data) >= 3 and data[:3] == _JPEG_MAGIC:
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == _PNG_MAGIC:
        return "image/png"
    if len(data) >= 6 and (data[:6] == _GIF_MAGIC or data[:6] == _GIF89_MAGIC):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 2 and data[:2] == _BMP_MAGIC:
        return "image/bmp"
    return None


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
