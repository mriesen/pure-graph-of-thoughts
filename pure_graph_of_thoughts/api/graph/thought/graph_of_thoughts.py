from dataclasses import dataclass
from typing import Self

from .thought_node import ThoughtNode
from ..graph import Graph
from ...state import State
from ...thought import Thought


@dataclass
class GraphOfThoughts(Graph[ThoughtNode]):
    """
    Represents a graph of thoughts.
    """

    @classmethod
    def from_init_state(cls, init_state: State) -> Self:
        """
        Creates a graph of thoughts from an initial state.
        :param init_state: initial state
        :return: created graph of thoughts
        """
        source = ThoughtNode.of(
                Thought(state=init_state)
        )
        return cls.from_source(source)
