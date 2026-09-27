"""Build the same platform identifiers used by the native pi user agent."""

import platform
import sys

__all__ = ["get_pi_user_agent"]


def get_pi_user_agent() -> str:
    if sys.platform in ("emscripten", "wasi"):
        return "pi (browser)"
    system = {"win32": "win32", "cygwin": "win32"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "x64",
        "amd64": "x64",
        "i386": "ia32",
        "i686": "ia32",
        "aarch64": "arm64",
        "arm64": "arm64",
        "ppc64le": "ppc64",
        "armv6l": "arm",
        "armv7l": "arm",
        "armv8l": "arm",
    }.get(machine, machine)
    return f"pi ({system} {platform.release()}; {architecture})"
