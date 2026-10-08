"""Validate an OpenCode source bundle before a Harbor evaluation."""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath


def _safe_link_target(member: tarfile.TarInfo) -> str:
    target = PurePosixPath(member.linkname)
    if not member.linkname or target.is_absolute():
        raise ValueError("source_bundle contains unsafe links")
    parts = list(PurePosixPath(member.name).parts[:-1]) if member.issym() else []
    for part in target.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError("source_bundle contains escaping links")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise ValueError("source_bundle contains unsafe links")
    return "/".join(parts)


def _resolve_archive_path(path: str, links: dict[str, str]) -> str:
    parts = path.split("/")
    resolved: list[str] = []
    hops = 0
    while parts:
        component = parts.pop(0)
        if component in ("", "."):
            continue
        if component == "..":
            if not resolved:
                raise ValueError("source_bundle contains escaping links")
            resolved.pop()
            continue
        resolved.append(component)
        current = "/".join(resolved)
        if current in links:
            hops += 1
            if hops > 32:
                raise ValueError("source_bundle contains cyclic links")
            parts = links[current].split("/") + parts
            resolved = []
    return "/".join(resolved)


def check_bundle(bundle: Path, checkout: Path | None = None) -> None:
    """Reject unsafe archives and, when provided, source/checkout mismatches."""
    path = Path(bundle).expanduser().resolve(strict=True)
    expected: dict[str, Path] = {}
    if checkout is not None:
        root = Path(checkout).expanduser().resolve(strict=True)
        tool_dir = root / "packages" / "opencode" / "src" / "tool"
        if not tool_dir.is_dir():
            raise ValueError("not an OpenCode source checkout")
        relpaths = [
            Path("package.json"),
            Path("bun.lock"),
            Path("packages/opencode/package.json"),
            Path("packages/opencode/src/index.ts"),
            *(
                file.relative_to(root)
                for file in tool_dir.rglob("*")
                if file.suffix in {".ts", ".txt"}
            ),
        ]
        for rel in relpaths:
            source = (root / rel).resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("OpenCode checkout contains unsafe source paths")
            expected[rel.as_posix()] = source

    required = {"package.json", "packages/opencode/src/index.ts"}
    names: set[str] = set()
    member_types: dict[str, bytes] = {}
    links: dict[str, str] = {}
    total_size = 0
    try:
        with tarfile.open(path, "r:*") as archive:
            for index, member in enumerate(archive):
                if index >= 200_000:
                    raise ValueError("source_bundle has too many entries")
                rel = PurePosixPath(member.name)
                if (
                    rel.is_absolute()
                    or ".." in rel.parts
                    or not (
                        member.isfile()
                        or member.isdir()
                        or member.issym()
                        or member.islnk()
                    )
                ):
                    raise ValueError("source_bundle contains unsafe entries")
                name = rel.as_posix()
                if name in names:
                    raise ValueError("source_bundle contains duplicate entries")
                names.add(name)
                member_types[name] = member.type
                if member.issym() or member.islnk():
                    links[name] = _safe_link_target(member)
                total_size += member.size
                if total_size > 8_000_000_000:
                    raise ValueError("source_bundle exceeds 8 GB uncompressed")
                if name in expected:
                    if not member.isfile() or member.size > 5_000_000:
                        raise ValueError(f"OpenCode source file is too large: {name}")
                    stream = archive.extractfile(member)
                    if stream is None or stream.read() != expected[name].read_bytes():
                        raise ValueError(
                            f"source_bundle differs from OpenCode checkout: {name}"
                        )
    except tarfile.TarError as exc:
        raise ValueError("source_bundle is not a readable tar archive") from exc

    for name, target in links.items():
        resolved = _resolve_archive_path(target, links)
        target_type = member_types.get(resolved)
        if target_type is None or (
            member_types[name] == tarfile.LNKTYPE
            and target_type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
        ):
            raise ValueError("source_bundle contains unresolved links")

    if not required <= names or not any(
        name.startswith("node_modules/") for name in names
    ):
        raise ValueError(
            "source_bundle needs OpenCode source and installed node_modules"
        )
    missing = expected.keys() - names
    if missing:
        raise ValueError(
            f"source_bundle is missing OpenCode source: {sorted(missing)[0]}"
        )
