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
        if state.shape == torch.Size([4, 4]):
            return torch.rand((), dtype=torch.float32, device=self.device)
        return torch.rand(state.shape[0], dtype=torch.float32, device=self.device)

    def update(self, state, reward, next_state):
        """Something."""
        pass