import json
import os
from pathlib import Path
from typing import Any

import toml
import yaml


def load_file(path: str | Path, **kwargs: Any) -> Any:
    """Load data from JSON/YAML/TOML/TXT/MD files."""
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r") as fp:
            return json.load(fp, **kwargs)
    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r") as fp:
            return yaml.load(fp, Loader=yaml.SafeLoader, **kwargs)
    if path.suffix == ".toml":
        with open(path, "r") as fp:
            return toml.load(fp, **kwargs)
    if path.suffix in {".txt", ".md"}:
        encoding = kwargs.pop("encoding", None)
        if kwargs:
            raise ValueError(f"Unsupported keyword arguments: {kwargs}")
        with open(path, "r", encoding=encoding) as fp:
            return fp.read()
    raise ValueError(f"Unsupported file extension: {path}")


def dump_file(path: str | Path, data: Any, **kwargs: Any) -> None:
    """Dump data to JSON/YAML/TOML/TXT/MD files."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    if path.suffix == ".json":
        with open(path, "w") as fp:
            json.dump(data, fp, **kwargs)
        return
    if path.suffix in {".yaml", ".yml"}:
        with open(path, "w") as fp:
            yaml.dump(data, fp, **kwargs)
        return
    if path.suffix == ".toml":
        data_str = json.dumps(data)
        new_data = json.loads(data_str)
        with open(path, "w") as fp:
            toml.dump(new_data, fp, **kwargs)
        return
    if path.suffix in {".txt", ".md"}:
        encoding = kwargs.pop("encoding", None)
        if kwargs:
            raise ValueError(f"Unsupported keyword arguments: {kwargs}")
        with open(path, "w", encoding=encoding) as fp:
            fp.write(data)
        return
    raise ValueError(f"Unsupported file extension: {path}")
