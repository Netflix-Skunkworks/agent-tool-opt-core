from typing import Any, Dict, TypeVar

from addict import Dict as AddictDict
from pydantic import BaseModel, ConfigDict

from agent_tool_opt_core.utils.hash_utils import get_pydantic_hash

T = TypeVar("T", bound=BaseModel)


class BaseModelNoExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


def update_pydantic_model_with_dict(
    model_instance: T, update_data: Dict[str, Any]
) -> T:
    """Return an updated BaseModel instance based on update data."""
    raw_data = AddictDict(model_instance.model_dump())
    raw_data.update(AddictDict(update_data))
    new_data = raw_data.to_dict()
    model_class = type(model_instance)
    return model_class.model_validate(new_data)


__all__ = [
    "BaseModelNoExtra",
    "get_pydantic_hash",
    "update_pydantic_model_with_dict",
]
