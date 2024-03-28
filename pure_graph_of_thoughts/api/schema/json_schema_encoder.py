import json
from enum import Enum
from typing import Any

from .schema import Schema


class JsonSchemaEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for schema serialization.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, Schema):
            return obj.to_dict()
        return super().default(obj)