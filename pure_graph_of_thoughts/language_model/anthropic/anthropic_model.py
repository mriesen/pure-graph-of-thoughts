from enum import Enum


class AnthropicModel(Enum):
    """
    Represents a specific Anthropic Claude model.
    """

    CLAUDE_HAIKU_4_5_20251001 = 'claude-haiku-4-5-20251001'
    CLAUDE_SONNET_4_6 = 'claude-sonnet-4-6'
    CLAUDE_OPUS_4_6 = 'claude-opus-4-6'

    @property
    def id(self) -> str:
        """The model ID"""
        return self.value


DEFAULT_ANTHROPIC_MODEL: AnthropicModel = AnthropicModel.CLAUDE_HAIKU_4_5_20251001
"""The default Anthropic model."""