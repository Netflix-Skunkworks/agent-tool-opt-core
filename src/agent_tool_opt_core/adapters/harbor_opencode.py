"""Harbor custom agent that runs a locally bundled OpenCode checkout from source.

This module is imported only by Harbor jobs. The bundle and Linux Bun binary
are explicit local inputs; no package manager or remote installation script is
run inside the benchmark sandbox. An overlay is uploaded before OpenCode starts.
"""

from __future__ import annotations

import tarfile
import tempfile
import shlex
from pathlib import Path, PurePosixPath

from pydantic import Field

from harbor.agents.installed.opencode import OpenCode, OpenCodeOptions
from harbor.environments.base import BaseEnvironment

_ROOT = "/opt/ato-opencode"
_SOURCE = f"{_ROOT}/source"
_BIN = f"{_ROOT}/bun"
_TOOL_DIR = f"{_SOURCE}/packages/opencode/src/tool"
_SHIM = """#!/bin/bash
args=()
for arg in "$@"; do
  case "$arg" in
    --dangerously-skip-permissions) ;;
    *) args+=("$arg") ;;
  esac
done
exec /opt/ato-opencode/bun run --cwd /opt/ato-opencode/source/packages/opencode --conditions=browser src/index.ts "${args[@]}"
"""


def _check_bundle(path: Path) -> None:
    if not path.is_file() or not tarfile.is_tarfile(path):
        raise ValueError("source_bundle must be a tar archive")
    required = {"package.json", "packages/opencode/src/index.ts"}
    names: set[str] = set()
    total_size = 0
    with tarfile.open(path, "r:*") as archive:
        for index, member in enumerate(archive):
            if index >= 200_000:
                raise ValueError("source_bundle has too many entries")
            rel = PurePosixPath(member.name)
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("source_bundle contains unsafe entries")
            total_size += member.size
            if total_size > 8_000_000_000:
                raise ValueError("source_bundle exceeds 8 GB uncompressed")
            names.add(rel.as_posix().removeprefix("./"))
    if not required <= names or not any(
        name.startswith("node_modules/") for name in names
    ):
        raise ValueError(
            "source_bundle needs OpenCode source and installed node_modules"
        )


class SourceOpenCodeOptions(OpenCodeOptions):
    source_bundle: str = Field(description="Local Linux OpenCode source tarball")
    bun_linux_binary: str = Field(description="Local Linux Bun executable")
    overlay_dir: str = Field(description="Local tool-description overlay directory")


class SourceOpenCode(OpenCode):
    """OpenCode from source, with a per-evaluation description overlay."""

    options_model = SourceOpenCodeOptions

    def __init__(
        self,
        *args,
        source_bundle: str,
        bun_linux_binary: str,
        overlay_dir: str,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            source_bundle=source_bundle,
            bun_linux_binary=bun_linux_binary,
            overlay_dir=overlay_dir,
            **kwargs,
        )
        self._bundle = Path(source_bundle).expanduser().resolve(strict=True)
        self._bun = Path(bun_linux_binary).expanduser().resolve(strict=True)
        self._overlay = Path(overlay_dir).expanduser().resolve(strict=True)
        _check_bundle(self._bundle)
        if not self._bun.is_file() or not self._overlay.is_dir():
            raise ValueError("Bun executable and overlay directory are required")
        with self._bun.open("rb") as binary:
            if binary.read(4) != b"\x7fELF":
                raise ValueError("bun_linux_binary must be a Linux ELF executable")

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("bash", "coreutils", "tar"))
        result = await environment.exec(
            command=f"mkdir -p {_SOURCE} {_TOOL_DIR}", user="root"
        )
        if result.return_code != 0:
            raise RuntimeError("failed to create OpenCode source directory")
        await environment.upload_file(
            source_path=self._bundle, target_path=f"{_ROOT}/source.tar"
        )
        await environment.upload_file(source_path=self._bun, target_path=_BIN)
        result = await environment.exec(
            command=f"chmod 755 {_BIN} && tar -xf {_ROOT}/source.tar -C {_SOURCE}",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError("failed to extract OpenCode source bundle")
        for path in sorted(self._overlay.rglob("*.txt")):
            if path.is_symlink() or not path.resolve().is_relative_to(self._overlay):
                raise ValueError("unsafe overlay file")
            relative = path.relative_to(self._overlay).as_posix()
            destination = f"{_TOOL_DIR}/{relative}"
            parent = str(PurePosixPath(destination).parent)
            result = await environment.exec(
                command=f"mkdir -p {shlex.quote(parent)}", user="root"
            )
            if result.return_code != 0:
                raise RuntimeError("failed to create OpenCode tool directory")
            await environment.upload_file(source_path=path, target_path=destination)
        with tempfile.TemporaryDirectory(prefix="ato-opencode-shim-") as directory:
            shim = Path(directory) / "opencode"
            shim.write_text(_SHIM, encoding="utf-8")
            await environment.upload_file(
                source_path=shim, target_path="/usr/local/bin/opencode"
            )
        result = await environment.exec(
            command="chmod 755 /usr/local/bin/opencode && opencode --version",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError("OpenCode from-source smoke test failed")
