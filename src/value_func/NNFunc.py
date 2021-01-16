"""Something."""
from . import ValueFunc


class NNFunc(ValueFunc):
    """Something."""

    def __init__(self, model, loss_func, optimizer, device):
        """Something."""
        self._device = device
        self._model = model.to(device)
        self._loss_func = loss_func
        self._optimizer = optimizer

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
        return self.model(state.flatten(-2, -1).float()).squeeze()

    def update(self, state, target):
        """Something."""
        value = self.evaluate(state)
        loss = self._loss_func(value, target.float())
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
