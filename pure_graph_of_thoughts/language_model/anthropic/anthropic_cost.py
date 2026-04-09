from dataclasses import dataclass
from typing import Sequence, Mapping

from .anthropic_model import AnthropicModel


@dataclass(frozen=True)
class AnthropicCost:
    """
    Represents the usage cost of an Anthropic model.
    """

    model: AnthropicModel
    """The Anthropic model"""

    input_token_cost: float
    """The cost per input token"""

    output_token_cost: float
    """The cost per output token"""

    currency: str = '$'
    """The currency"""


_anthropic_costs: Sequence[AnthropicCost] = [
    AnthropicCost(
            model=AnthropicModel.CLAUDE_HAIKU_4_5_20251001,
            input_token_cost=1.0 / 1_000_000,
            output_token_cost=5.0 / 1_000_000
    ),
    AnthropicCost(
            model=AnthropicModel.CLAUDE_SONNET_4_6,
            input_token_cost=3.0 / 1_000_000,
            output_token_cost=15.0 / 1_000_000
    ),
    AnthropicCost(
            model=AnthropicModel.CLAUDE_OPUS_4_6,
            input_token_cost=5.0 / 1_000_000,
            output_token_cost=25.0 / 1_000_000
    )
]

anthropic_costs_by_model: Mapping[AnthropicModel, AnthropicCost] = {
    cost.model: cost for cost in _anthropic_costs
}
"""The cost of the Anthropic models by model."""