"""Shell selection and process-tree helpers from ``utils/shell.ts``."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_bin_dir


@dataclass
class ShellConfig:
    shell: str
    args: list[str] = field(default_factory=list)
    command_transport: str = "argv"


def _bash_config(shell: str) -> ShellConfig:
    normalized = shell.replace("/", "\\").lower()
    legacy_wsl = bool(re.fullmatch(r"[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe", normalized))
    return ShellConfig(shell, ["-s"] if legacy_wsl else ["-c"], "stdin" if legacy_wsl else "argv")


def get_shell_config(custom_shell_path: str | None = None) -> ShellConfig:
    if custom_shell_path:
        if Path(custom_shell_path).exists():
            return _bash_config(custom_shell_path)
        raise RuntimeError(f"Custom shell path not found: {custom_shell_path}")
    if os.name == "nt":
        candidates = [
            str(Path(value) / "Git" / "bin" / "bash.exe")
            for value in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"))
            if value
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return _bash_config(candidate)
        on_path = shutil.which("bash.exe")
        if on_path:
            return _bash_config(on_path)
        raise RuntimeError("No bash shell found. Install Git Bash or set shellPath in settings.json.")
    if Path("/bin/bash").exists():
        return _bash_config("/bin/bash")
    on_path = shutil.which("bash")
    return _bash_config(on_path) if on_path else ShellConfig("sh", ["-c"])


def get_shell_env() -> dict[str, str]:
    env = dict(os.environ)
    bin_dir = get_bin_dir()
    path_key = next((key for key in env if key.lower() == "path"), "PATH")
    current_path = env.get(path_key, "")
    if bin_dir not in current_path.split(os.pathsep):
        env[path_key] = os.pathsep.join(value for value in (bin_dir, current_path) if value)
    return env


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        taskkill = str(Path(system_root) / "System32" / "taskkill.exe")
        try:
            subprocess.Popen(
                [taskkill, "/F", "/T", "/PID", str(pid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS,
            )
        except OSError:
            pass
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


__all__ = ["ShellConfig", "get_shell_config", "get_shell_env", "kill_process_tree"]
