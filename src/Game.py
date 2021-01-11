"""Code Implementing Game Logic."""
import torch
from numba import guvectorize


LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3


def _can_advance(state, can):
    can[0] = False
    for x in range(state.shape[0]):
        for y1 in range(state.shape[1] - 1):
            for y2 in range(y1 + 1, state.shape[1]):
                if state[x, y2] == 0:
                    continue
                elif state[x, y1] == 0:
                    can[0] = True
                else:
                    if state[x, y1] == state[x, y2]:
                        can[0] = True
                    break


def _advance(state, reward):
    reward[0] = 0
    for x in range(state.shape[0]):
        for y1 in range(state.shape[1] - 1):
            for y2 in range(y1 + 1, state.shape[1]):
                if state[x, y2] == 0:
                    continue
                elif state[x, y1] == 0:
                    state[x, y1] = state[x, y2]
                    state[x, y2] = 0
                else:
                    if state[x, y1] == state[x, y2]:
                        state[x, y1] += 1
                        state[x, y2] = 0
                        reward[0] += 2 ** state[x, y1]
                    break


def _add_tile(state, random1, random2):
    num_zero = 0
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y] == 0:
                num_zero += 1
    if num_zero != 0:
        nth_zero = int(num_zero * random1[0])
        count = 0
        for _x in range(state.shape[0]):
            for _y in range(state.shape[1]):
                if state[_x, _y] == 0:
                    if count == nth_zero:
                        x, y = _x, _y
                    count += 1
        state[x, y] = 1 if random2[0] < 0.9 else 2


class Game(object):
    """Something."""

    def __init__(self, batch_size=100, board_size=4, history=None,
                 device=torch.device("cpu"), target=None):
        """Something."""
        self._batch_size = batch_size
        self._board_size = board_size
        self._history = history
        self._device = device
        if target is None:
            self._target = self.device.type
        else:
            self._target = target
        self._can_advance = guvectorize(['void(u1[:,:], u1[:])'], '(n,n)->()',
                                        target=self.target,
                                        nopython=True)(_can_advance)
        self._advance = guvectorize(['void(u1[:,:], i4[:])'], '(n,n)->()',
                                    target=self.target,
                                    nopython=True)(_advance)
        self._add_tile = guvectorize(['void(u1[:,:], f4[:], f4[:])'],
                                     '(n,n),(),()', target=self.target,
                                     nopython=True)(_add_tile)
        self._state = torch.zeros((self.batch_size, self.board_size,
                                   self.board_size), device=self.device,
                                  dtype=torch.uint8)
        self._score = torch.zeros(self.batch_size, device=self.device,
                                  dtype=torch.int32)
        self._rotation = torch.zeros(self.batch_size, device=self.device,
                                     dtype=torch.int32)
        self.add_tile()
        self.add_tile()

    @property
    def batch_size(self):
        """Something."""
        return self._batch_size

    @property
    def board_size(self):
        """Something."""
        return self._board_size

    @property
    def history(self):
        """Something"""
        return self._history

    @property
    def device(self):
        """Something."""
        return self._device

    @property
    def target(self):
        """Something."""
        return self._target

    @property
    def state(self):
        """Something."""
        return self._state

    @property
    def rotation(self):
        """Something."""
        return self._rotation

    @property
    def score(self):
        """Something."""
        return self._score

    def _device_check(self, tensor):
        if self.device.type == 'cpu':
            return tensor.numpy()
        return tensor

    def new_game(self):
        """Something."""
        self._state *= 0
        self._score *= 0
        self.add_tile()
        self.add_tile()

    def rotations(self):
        """Something."""
        rotations = [self._state.rot90(i, (-2, -1)) for i in range(0, 4)]
        return torch.stack(rotations, -3)

    def valid_rotations(self):
        """Somthing."""
        return self._can_advance(self.rotations())

    def game_over(self, **kwargs):
        return self.valid_rotations().any(**kwargs).logical_not()

    def rotate(self, k):
        """Something."""
        self._state = self.rotations()[torch.arange(self.batch_size), k]
        self._rotation = (self._rotation + k) % 4

    def can_advance(self):
        """Something."""
        can = torch.zeros(self.batch_size, device=self.device,
                          dtype=torch.uint8)
        self._can_advance(self._state, out=self._device_check(can))
        return can

    def advance(self):
        """Something."""
        if self.history is not None:
            self.history.append(self)
        reward = torch.zeros(self.batch_size, device=self.device,
                             dtype=torch.int32)
        self._advance(self._state, out=self._device_check(reward))
        self._score += reward
        return reward

    def add_tile(self):
        """Something."""
        random1 = torch.rand(self.batch_size, device=self.device,
                             dtype=torch.float32)
        random2 = torch.rand(self.batch_size, device=self.device,
                             dtype=torch.float32)
        self._add_tile(self._state, random1, random2)
