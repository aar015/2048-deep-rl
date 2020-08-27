"""Something."""
from abc import ABC, abstractmethod


class ValueFunc(ABC):
    """Something."""

    @abstractmethod
    def evaluate(self, state):
        """Something."""
        pass

    @abstractmethod
    def update(self, state, reward, next_state):
        """Something."""
        pass