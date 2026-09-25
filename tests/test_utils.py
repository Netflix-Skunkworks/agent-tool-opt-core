"""Tests for utils (hashing, pydantic, IO) and the DRAFT description adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_tool_opt_core.utils.hash_utils import get_dict_hash


# ---------------------------------------------------------------------------
# hash_utils
# ---------------------------------------------------------------------------


def test_get_dict_hash_is_stable_and_order_insensitive():
    a = get_dict_hash({"b": 1, "a": 2})
    b = get_dict_hash({"a": 2, "b": 1})
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_get_dict_hash_changes_with_content():
    assert get_dict_hash({"a": 1}) != get_dict_hash({"a": 2})


def test_get_dict_hash_handles_non_json_values():
    # default=str lets non-serializable values (e.g. Path) hash without error
    assert len(get_dict_hash({"p": Path("/x"), "n": 1})) == 64


# ---------------------------------------------------------------------------
# pydantic_utils
# ---------------------------------------------------------------------------


def test_update_pydantic_model_with_dict_and_hash():
    pytest.importorskip("addict")
    from pydantic import BaseModel

    from agent_tool_opt_core.utils.hash_utils import get_pydantic_hash
    from agent_tool_opt_core.utils.pydantic_utils import update_pydantic_model_with_dict

    class M(BaseModel):
        x: int
        y: str = "default"

    m = M(x=1)
    m2 = update_pydantic_model_with_dict(m, {"x": 5})
    assert m2.x == 5
    assert m2.y == "default"
    assert m.x == 1  # original is not mutated
    assert get_pydantic_hash(m) != get_pydantic_hash(m2)


# ---------------------------------------------------------------------------
# io_utils
# ---------------------------------------------------------------------------


def test_io_roundtrip_structured_and_text(tmp_path):
    pytest.importorskip("yaml")
    pytest.importorskip("toml")
    from agent_tool_opt_core.utils.io_utils import dump_file, load_file

    data = {"a": 1, "b": ["x", "y"]}
    for ext in (".json", ".yaml", ".toml"):
        p = tmp_path / f"f{ext}"
        dump_file(p, data)
        assert load_file(p) == data

    txt = tmp_path / "note.txt"
    dump_file(txt, "hello")
    assert load_file(txt) == "hello"


def test_io_unsupported_extension_raises(tmp_path):
    from agent_tool_opt_core.utils.io_utils import load_file

    (tmp_path / "f.xyz").write_text("x")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_file(tmp_path / "f.xyz")


# (draft/adapter.apply_draft_descriptions was retired with the old interface.)
