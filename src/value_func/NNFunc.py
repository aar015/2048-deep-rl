"""Something."""
from . import ValueFunc
import torch


class NNFunc(ValueFunc):
    """Something."""

    def __init__(self, model, device):
        """Something."""
        self._device = device
        self._model = model.to(device)

    @property
    def model(self):
        """Something."""
        return self._model

    @property
    def device(self):
        """Something."""
        return self._device

    def evaluate(self, state):
        """Something."""
        if state.shape == torch.Size([4, 4]):
            return self.model(state.flatten().float())
        return self.model(state.flatten(1).float()).flatten()

    def update(self, state, reward, next_state, loss_func, optimizer):
        """Something."""
        value = self.evaluate(state)
        target = reward + self.evaluate(next_state).detach()
        loss = loss_func(value, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
