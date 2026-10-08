"""Optional upstream Bubblewrap confinement for the Pi subprocess on Linux.

The sandbox shares the host network for provider calls. It exposes system
runtime files read-only, the workspace read-only with individual editable files
writable, and a separate writable session directory. It does not isolate the
benchmark or the Python candidate validator.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping


PROVIDER_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "XAI_API_KEY",
        "CEREBRAS_API_KEY",
        "ZAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
    }
)
_RUNTIME_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib64")
_SYSTEM_FILES = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/ssl/certs",
    "/etc/pki/tls/certs",
)
_SANDBOX_PATH = "/usr/local/bin:/usr/bin:/bin"


def validate_sandbox_options(mode: str, env_names: tuple[str, ...]) -> None:
    if mode not in {"none", "bubblewrap"}:
        raise ValueError("Pi sandbox must be 'none' or 'bubblewrap'")
    if len(env_names) > len(PROVIDER_ENV_KEYS) or any(
        not isinstance(name, str) or name not in PROVIDER_ENV_KEYS for name in env_names
    ):
        raise ValueError("unsupported Pi sandbox environment variable")
    if env_names and mode != "bubblewrap":
        raise ValueError("Pi sandbox environment variables require Bubblewrap mode")


def bubblewrap_runtime(binary: str, source_env: Mapping[str, str]) -> tuple[str, str]:
    """Resolve the explicitly requested runtime; never fall back to host execution."""
    if sys.platform != "linux":
        raise RuntimeError("Pi Bubblewrap mode requires Linux")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("Pi Bubblewrap mode requires the upstream bwrap executable")
    pi = shutil.which(binary, path=source_env.get("PATH", os.defpath))
    if pi is None:
        raise RuntimeError("Pi executable was not found")
    runtime_roots = [Path(name).resolve() for name in _RUNTIME_DIRS]
    resolved = Path(pi).resolve(strict=True)
    if not any(resolved.is_relative_to(root) for root in runtime_roots):
        raise RuntimeError(
            "Pi Bubblewrap mode requires Pi installed under /usr or /usr/local; "
            "home-directory installations are not exposed"
        )
    node = shutil.which("node", path=_SANDBOX_PATH)
    if node is None:
        raise RuntimeError("Pi Bubblewrap mode requires Node.js on the system PATH")
    if not any(
        Path(node).resolve(strict=True).is_relative_to(root) for root in runtime_roots
    ):
        raise RuntimeError(
            "Pi Bubblewrap mode requires Node.js installed under /usr or /usr/local"
        )
    return str(Path(bwrap).resolve(strict=True)), str(resolved)


def bubblewrap_command(
    command: list[str],
    workspace: Path,
    editable: tuple[str, ...],
    source_env: Mapping[str, str],
    env_names: tuple[str, ...],
) -> tuple[list[str], dict[str, str]]:
    """Build an argument vector and minimal environment without shell interpolation."""
    validate_sandbox_options("bubblewrap", env_names)
    bwrap, pi = bubblewrap_runtime(command[0], source_env)
    workspace = Path(workspace)
    if workspace.is_symlink():
        raise ValueError("Pi sandbox workspace must not be a symlink")
    workspace = workspace.resolve(strict=True)
    sessions = workspace.parent / ".pi-session"
    if (
        not workspace.is_dir()
        or sessions.is_symlink()
        or not sessions.is_dir()
        or sessions.resolve(strict=True).parent != workspace.parent
    ):
        raise ValueError(
            "Pi sandbox requires a workspace and sibling session directory"
        )
    if any(workspace.is_relative_to(Path(name).resolve()) for name in _RUNTIME_DIRS):
        raise ValueError("Pi scratch must be outside the exposed system runtime trees")

    environment = {
        "PATH": _SANDBOX_PATH,
        "HOME": "/home/pi",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "PI_TELEMETRY": "0",
    }
    for name in env_names:
        value = source_env.get(name)
        if not isinstance(value, str) or not 1 <= len(value) <= 65_536 or "\0" in value:
            raise ValueError(
                f"requested Pi sandbox environment variable is unset: {name}"
            )
        environment[name] = value

    argv = [
        bwrap,
        "--unshare-all",
        "--unshare-user",
        "--share-net",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for name in (*_RUNTIME_DIRS, *_SYSTEM_FILES):
        source = Path(name)
        if source.exists():
            argv.extend(("--ro-bind", str(source.resolve(strict=True)), name))
    argv.extend(
        (
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/home/pi",
            "--ro-bind",
            str(workspace),
            "/workspace",
            "--bind",
            str(sessions),
            "/sessions",
        )
    )
    for name in editable:
        relative = PurePosixPath(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe Pi sandbox editable path")
        source = workspace / name
        resolved = source.resolve(strict=True)
        if (
            not resolved.is_relative_to(workspace)
            or resolved != source
            or not resolved.is_file()
            or resolved.stat().st_nlink != 1
        ):
            raise ValueError(
                "Pi sandbox editable files must be ordinary workspace files"
            )
        argv.extend(
            ("--bind", str(resolved), str(PurePosixPath("/workspace") / relative))
        )

    # Keep credentials in the subprocess environment, never in the argument list.
    argv.extend(("--chdir", "/workspace", "--", pi, *command[1:]))
    return argv, environment


def add_pi_sandbox_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pi-sandbox",
        choices=("none", "bubblewrap"),
        default="none",
        help="Optional Linux filesystem sandbox for the Pi optimizer only",
    )
    parser.add_argument(
        "--pi-sandbox-env",
        action="append",
        choices=sorted(PROVIDER_ENV_KEYS),
        default=[],
        metavar="NAME",
        help=(
            "Forward a named provider variable into Pi's sandbox; repeat as needed. "
            "Allowed names: " + ", ".join(sorted(PROVIDER_ENV_KEYS))
        ),
    )


def pi_sandbox_kwargs(
    mode: str, env_names: tuple[str, ...], *, optimizer: str, enabled: bool
) -> dict:
    validate_sandbox_options(mode, env_names)
    if mode == "none":
        return {}
    if optimizer != "pi" or not enabled:
        raise ValueError("Pi sandbox options require an enabled Pi optimizer")
    return {"sandbox": mode, "sandbox_env": env_names}
