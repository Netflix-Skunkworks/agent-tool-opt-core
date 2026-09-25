import hashlib
import json

from pydantic import BaseModel


def get_dict_hash(obj: dict) -> str:
    """Generate a stable SHA256 hash for a dictionary."""
    hash_string = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def get_pydantic_hash(obj: BaseModel) -> str:
    """Generate a stable SHA256 hash for a pydantic model."""
    return get_dict_hash(obj.model_dump())
