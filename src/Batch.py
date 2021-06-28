"""Code Implementing Batch Game Logic."""
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


def _add_tile(state, random1, random2, success):
    num_zero = 0
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y] == 0:
                num_zero += 1
    if num_zero != 0:
        nth_zero = int(num_zero * random1[0])
        x, y, count = 0, 0, 0
        for _x in range(state.shape[0]):
            for _y in range(state.shape[1]):
                if state[_x, _y] == 0:
                    if count == nth_zero:
                        x, y = _x, _y
                    count += 1
        state[x, y] = 1 if random2[0] < 0.9 else 2
    success[0] = num_zero != 0


class Batch(object):
    """Something."""

    def __init__(
        self, size=1000, board_size=4, device=None, target=None, log=False
    ):
        """Initialize 2048 batch player with random initial boards."""
        # Set batch and board size
        self._size = size
        self._board_size = board_size

        # Determine pytorch device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self._device = device

        # Initialize log
        if log:
            pass
        else:
            self._log = None

        # Determine numba target
        if target is None:
            self._target = self.device.type
        else:
            self._target = target

        # Define numba accelerated functions
        self._can_advance = guvectorize(
            ['void(u1[:,:], u1[:])'],
            '(n,n)->()',
            target=self.target,
            nopython=True,
        )(_can_advance)
        self._advance = guvectorize(
            ['void(u1[:,:], i4[:])'],
            '(n,n)->()',
            target=self.target,
            nopython=True,
        )(_advance)
        self._add_tile = guvectorize(
            ['void(u1[:,:], f4[:], f4[:], u1[:])'],
            '(n,n),(),()->()',
            target=self.target,
            nopython=True,
        )(_add_tile)

        # Initialize batch state data
        self._states = torch.zeros(
            (self.size, self.board_size, self.board_size),
            device=self.device,
            dtype=torch.uint8)
        self._scores = torch.zeros(
            self.size,
            device=self.device,
            dtype=torch.int32)
        self._rotations = torch.zeros(
            self.size,
            device=self.device,
            dtype=torch.int8)

        # Add first two tiles
        self.add_tile()
        self.add_tile()

    @property
    def size(self):
        """Get batch size."""
        return self._size

    @property
    def board_size(self):
        """Get board size."""
        return self._board_size

    @property
    def log(self):
        """Get batch log."""
        return self._log

    @property
    def device(self):
        """Get pytorch device."""
        return self._device

    @property
    def target(self):
        """Get numba target."""
        return self._target

    @property
    def states(self):
        """Get states of games in batch."""
        return self._states

    @property
    def scores(self):
        """Get scores of games in batch."""
        return self._scores

    @property
    def rotations(self):
        """Get rotations of games in batch."""
        return self._rotations

    def _device_check(self, tensor):
        """Cast tensor for input into numba."""
        if self.device.type == 'cpu':
            return tensor.numpy()
        return tensor

    def new_game(self):
        """Reset batch state to new game."""
        self._states *= 0
        self._scores *= 0
        self._rotations *= 0
        self.add_tile()
        self.add_tile()

    def sym_rotations(self):
        """List rotational symmetries."""
        rotations = [self.states.rot90(i, (-2, -1)) for i in range(0, 4)]
        return torch.stack(rotations, -3)

    def sym_rotations_advance(self):
        """Determine which rotational symmetries that can advance."""
        can = torch.zeros(
            (self.size, 4), device=self.device, dtype=torch.uint8)
        self._can_advance(self.sym_rotations(), out=self._device_check(can))
        return can

    def game_over(self, **kwargs):
        """Determine which games have finished."""
        return self.sym_rotations_advance().any(**kwargs).logical_not()

    def rotate(self, k):
        """Rotate game states 90 degrees counterclockwise k times."""
        self._states = self.sym_rotations()[torch.arange(self.size), k]
        self._rotations = ((self.rotations + k) % 4).type(torch.uint8)

    def can_advance(self):
        """Determine which games can advance."""
        can = torch.zeros(self.size, device=self.device, dtype=torch.uint8)
        self._can_advance(self.states, out=self._device_check(can))
        return can

    def advance(self):
        """Advance games."""
        if self.log is not None:
            self.log.append(self)
        rewards = torch.zeros(self.size, device=self.device, dtype=torch.int32)
        self._advance(self._states, out=self._device_check(rewards))
        self._scores += rewards
        return rewards

    def add_tile(self):
        """Add tile to game states."""
        rand1 = torch.rand(self.size, device=self.device, dtype=torch.float32)
        rand2 = torch.rand(self.size, device=self.device, dtype=torch.float32)
        ok = torch.zeros(self.size, device=self.device, dtype=torch.uint8)
        self._add_tile(self._states, rand1, rand2, out=self._device_check(ok))
        return ok
