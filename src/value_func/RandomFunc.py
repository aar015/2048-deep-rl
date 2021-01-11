"""Something."""
from . import ValueFunc
import torch


class RandomFunc(ValueFunc):
    """Something."""

    def __init__(self, device):
        """Something."""
        self._device = device

    @property
    def device(self):
        """Something."""
        return self._device

    def evaluate(self, state):
        """Something."""
        return torch.rand(state.shape[:-2], dtype=torch.float32,
                          device=self.device)

    def update(self, state, target):
        """Something."""
        pass
