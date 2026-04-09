import json
import logging
from typing import Self, cast

import backoff
import anthropic
from anthropic import RateLimitError

from .anthropic_cost import AnthropicCost, anthropic_costs_by_model
from .anthropic_model import AnthropicModel, DEFAULT_ANTHROPIC_MODEL
from .anthropic_usage import AnthropicUsage
from ...api.language_model import LanguageModelException
from ...api.language_model.language_model import LanguageModel
from ...api.language_model.prompt import Prompt
from ...api.state import State

_JSON_PREFIX = '```json'
_JSON_SUFFIX = '```'
_SYSTEM_PROMPT = 'Only answer in JSON with the schema described by the examples.'

class AnthropicLanguageModel(LanguageModel):
    """
    The Anthropic Claude language model.
    """

    _model: AnthropicModel
    _cost: AnthropicCost
    _temperature: float = 1.0
    _max_tokens: int = 1536
    _n_total_input_tokens: int = 0
    _n_total_output_tokens: int = 0
    _client: anthropic.Anthropic
    _total_cost: float = 0
    _logger: logging.Logger

    @property
    def usage(self) -> AnthropicUsage:
        return AnthropicUsage(
                model=self._model,
                n_input_tokens=self._n_total_input_tokens,
                n_output_tokens=self._n_total_output_tokens,
                total_cost=self._total_cost,
                currency=self._cost.currency
        )

    def __init__(self, api_key: str, model: AnthropicModel = DEFAULT_ANTHROPIC_MODEL) -> None:
        """
        Initializes a new AnthropicLanguageModel instance.
        :param api_key: Anthropic API key
        :param model: Anthropic model to use
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._cost = anthropic_costs_by_model[self._model]
        self._client = anthropic.Anthropic(api_key=api_key)

    @backoff.on_exception(
            backoff.expo, RateLimitError, logger=Self.__class__.__name__, max_time=30, max_tries=3, factor=10
    )
    def prompt(self, prompt: Prompt, state: State) -> State:
        """
        Queries the Anthropic API and returns the output state.
        :param prompt: prompt to use
        :param state: input state to use
        :return: output state
        """
        self._logger.debug('Calling Anthropic API with prompt %s and state %s', prompt, state)
        response = self._client.messages.create(
                model=self._model.id,
                system=_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt.for_input(state)}],
                temperature=self._temperature,
                max_tokens=self._max_tokens
        )
        n_input_tokens = response.usage.input_tokens
        n_output_tokens = response.usage.output_tokens
        self._n_total_input_tokens += n_input_tokens
        self._n_total_output_tokens += n_output_tokens
        delta_cost = (
                self._cost.input_token_cost * n_input_tokens
                + self._cost.output_token_cost * n_output_tokens
        )
        self._add_cost(delta_cost)
        self._logger.debug('Response Anthropic: %s', response)
        self._logger.debug(
                'Cost delta / total: %s %s / %s %s',
                delta_cost, self._cost.currency,
                self._total_cost, self._cost.currency
        )
        if response.content and hasattr(response.content[0], 'text'):
            content: str = response.content[0].text  # type: ignore[union-attr]
            if content is not None:
                if content.startswith(_JSON_PREFIX):
                    content = content.removeprefix(_JSON_PREFIX)
                if content.endswith(_JSON_SUFFIX):
                    content = content.removesuffix(_JSON_SUFFIX)
            return cast(State, json.loads(content.strip()))
        raise LanguageModelException('Response content is None or not a text block')

    def _add_cost(self, delta_cost: float) -> None:
        self._total_cost += delta_cost