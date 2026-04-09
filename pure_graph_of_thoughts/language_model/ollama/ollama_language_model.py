import json
import logging
from typing import cast

from ollama import Client, ChatResponse

from ...api.language_model import LanguageModelException
from ...api.language_model.language_model import LanguageModel
from ...api.language_model.prompt import Prompt
from ...api.state import State

DEFAULT_OLLAMA_HOST = 'http://localhost:11434'
"""The default Ollama host."""


class OllamaLanguageModel(LanguageModel):
    """
    A local language model served via Ollama.
    """

    _model: str
    _temperature: float = 1.0
    _client: Client
    _logger: logging.Logger

    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_HOST) -> None:
        """
        Initializes a new OllamaLanguageModel instance.
        :param model: Ollama model name (e.g. 'llama3', 'mistral', 'phi3')
        :param host: Ollama server host URL
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._client = Client(host=host)

    def prompt(self, prompt: Prompt, state: State) -> State:
        """
        Queries the local Ollama model and returns the output state.
        :param prompt: prompt to use
        :param state: input state to use
        :return: output state
        """
        self._logger.debug('Calling Ollama model %s with prompt %s and state %s', self._model, prompt, state)
        response: ChatResponse = self._client.chat(
                model=self._model,
                messages=[{'role': 'user', 'content': prompt.for_input(state)}],
                format='json',
                options={'temperature': self._temperature}
        )
        self._logger.debug('Response Ollama: %s', response)
        content = response.message.content
        if content is not None:
            return cast(State, json.loads(content.strip()))
        raise LanguageModelException('Response content is None')