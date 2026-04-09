from dataclasses import dataclass
from typing import Dict, Any, Self

from .anthropic_model import AnthropicModel
from ...api.schema import Schema


@dataclass(frozen=True)
class AnthropicUsage(Schema):
    """
    Represents the usage of an Anthropic model.
    """

    model: AnthropicModel
    """The used model"""

    n_input_tokens: int
    """The number of input tokens"""

    n_output_tokens: int
    """The number of output tokens"""

    total_cost: float
    """The total cost"""

    currency: str
    """The currency of the cost"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.value,
            'n_input_tokens': self.n_input_tokens,
            'n_output_tokens': self.n_output_tokens,
            'total_cost': self.total_cost,
            'currency': self.currency
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(
                model=AnthropicModel(data['model']),
                n_input_tokens=data['n_input_tokens'],
                n_output_tokens=data['n_output_tokens'],
                total_cost=data['total_cost'],
                currency=data['currency']
        )