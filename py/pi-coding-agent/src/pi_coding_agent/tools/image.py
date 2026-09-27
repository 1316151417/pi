"""Image sniffing, conversion, and resizing for ``read.ts``."""

from __future__ import annotations

import asyncio
import base64
import io
import math
from dataclasses import dataclass

from PIL import Image, ImageOps

IMAGE_TYPE_SNIFF_BYTES = 4100
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DEFAULT_MAX_BASE64_BYTES = 4.5 * 1024 * 1024


def detect_supported_image_mime_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return None if len(data) > 3 and data[3] == 0xF7 else "image/jpeg"
    if data.startswith(_PNG_SIGNATURE):
        if len(data) < 16 or int.from_bytes(data[8:12], "big") != 13 or data[12:16] != b"IHDR":
            return None
        offset = len(_PNG_SIGNATURE)
        while offset + 8 <= len(data):
            length = int.from_bytes(data[offset:offset + 4], "big")
            kind = data[offset + 4:offset + 8]
            if kind == b"acTL":
                return None
            if kind == b"IDAT":
                break
            next_offset = offset + 8 + length + 4
            if next_offset <= offset or next_offset > len(data):
                break
            offset = next_offset
        return "image/png"
    if data.startswith(b"GIF"):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM") and len(data) >= 26:
        declared = int.from_bytes(data[2:6], "little")
        pixel_offset = int.from_bytes(data[10:14], "little")
        header_size = int.from_bytes(data[14:18], "little")
        if declared and declared < 26 or pixel_offset < 14 + header_size or declared and pixel_offset >= declared:
            return None
        if header_size == 12:
            planes = int.from_bytes(data[22:24], "little")
            bits = int.from_bytes(data[24:26], "little")
        elif 40 <= header_size <= 124 and len(data) >= 30:
            planes = int.from_bytes(data[26:28], "little")
            bits = int.from_bytes(data[28:30], "little")
        else:
            return None
        return "image/bmp" if planes == 1 and bits in (1, 4, 8, 16, 24, 32) else None
    return None


async def detect_supported_image_mime_type_from_file(path: str) -> str | None:
    def read_prefix() -> bytes:
        with open(path, "rb") as stream:
            return stream.read(IMAGE_TYPE_SNIFF_BYTES)
    return detect_supported_image_mime_type(await asyncio.to_thread(read_prefix))


@dataclass(frozen=True)
class ProcessImageOptions:
    auto_resize_images: bool = True
    max_width: int = 2000
    max_height: int = 2000
    max_bytes: float = _DEFAULT_MAX_BASE64_BYTES
    jpeg_quality: int = 80


@dataclass(frozen=True)
class ProcessImageResult:
    ok: bool
    data: str = ""
    mime_type: str = ""
    hints: tuple[str, ...] = ()
    message: str = ""


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _image_bytes(image: Image.Image, kind: str, quality: int = 80) -> bytes:
    output = io.BytesIO()
    if kind == "JPEG":
        image.convert("RGB").save(output, format="JPEG", quality=quality)
    else:
        image.save(output, format="PNG")
    return output.getvalue()


def _process_image_sync(data: bytes, mime_type: str, options: ProcessImageOptions) -> ProcessImageResult:
    source = mime_type.split(";", 1)[0].strip().lower()
    supported = {
        "image/png": "image/png", "image/jpeg": "image/jpeg", "image/jpg": "image/jpeg",
        "image/gif": "image/gif", "image/webp": "image/webp",
    }
    normalized = supported.get(source)
    converted_from: str | None = None
    if normalized is None:
        try:
            with Image.open(io.BytesIO(data)) as opened:
                data = _image_bytes(ImageOps.exif_transpose(opened), "PNG")
        except Exception:
            return ProcessImageResult(False, message="[Image omitted: could not be converted to a supported inline image format.]")
        normalized = "image/png"
        converted_from = source
    hints: list[str] = []
    if converted_from is not None and converted_from != normalized:
        hints.append(f"[Image converted from {converted_from} to {normalized}.]")
    if not options.auto_resize_images:
        return ProcessImageResult(True, _base64(data), normalized, tuple(hints))
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            original_width, original_height = image.size
            if (original_width <= options.max_width and original_height <= options.max_height
                    and math.ceil(len(data) / 3) * 4 < options.max_bytes):
                return ProcessImageResult(True, _base64(data), normalized, tuple(hints))
            width, height = original_width, original_height
            if width > options.max_width:
                height = math.floor(height * options.max_width / width + 0.5)
                width = options.max_width
            if height > options.max_height:
                width = math.floor(width * options.max_height / height + 0.5)
                height = options.max_height
            width, height = max(1, width), max(1, height)
            qualities = list(dict.fromkeys((options.jpeg_quality, 85, 70, 55, 40)))
            while True:
                resized = image.resize((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
                for kind, quality in [("PNG", 0), *(("JPEG", value) for value in qualities)]:
                    candidate = _base64(_image_bytes(resized, kind, quality))
                    if len(candidate.encode("utf-8")) < options.max_bytes:
                        scale = original_width / width
                        hints.append(
                            f"[Image: original {original_width}x{original_height}, displayed at "
                            f"{width}x{height}. Multiply coordinates by {scale:.2f} to map to original image.]"
                        )
                        return ProcessImageResult(
                            True, candidate, "image/png" if kind == "PNG" else "image/jpeg", tuple(hints),
                        )
                if width == 1 and height == 1:
                    break
                next_width = max(1, math.floor(width * 0.75))
                next_height = max(1, math.floor(height * 0.75))
                if next_width == width and next_height == height:
                    break
                width, height = next_width, next_height
    except Exception:
        pass
    return ProcessImageResult(False, message="[Image omitted: could not be resized below the inline image size limit.]")


async def process_image(
    data: bytes, mime_type: str, options: ProcessImageOptions | None = None,
) -> ProcessImageResult:
    return await asyncio.to_thread(_process_image_sync, data, mime_type, options or ProcessImageOptions())


__all__ = [
    "IMAGE_TYPE_SNIFF_BYTES", "ProcessImageOptions", "ProcessImageResult",
    "detect_supported_image_mime_type", "detect_supported_image_mime_type_from_file",
    "process_image",
]
