"""Define state of 2048 game."""
from .app import SETTINGS
from functools import partial, wraps
from jax import jit, vmap, random
from jax import numpy as jnp
from numba import guvectorize
from numba.core.errors import NumbaDeprecationWarning
from pydantic.types import constr
from typing import List
from warnings import simplefilter

simplefilter('ignore', category=NumbaDeprecationWarning)

hex = partial(int, base=16)
State = constr(
    strip_whitespace=True, to_lower=True,
    min_length=16, max_length=16, regex='^[a-f0-9]*$'
)


class States(object):
    """Batch of 2048 board states."""

    def __init__(
        self, key=None, n=100, string=None, small=None, medium=None, large=None
    ):
        """Initialize states."""
        self._n = None
        self._string = None
        self._small = None
        self._medium = None
        self._large = None
        self._valid = None
        self._validate = None
        self._next = None
        self._reward = None
        self._rotations = None
        self._terminal = None
        self._live = None

        if key is not None:
            self._n = n
            keys = random.split(key, n)
            self._large = _init_state(keys)
        elif string is not None:
            self._string = string
        elif small is not None:
            self._small = small
        elif medium is not None:
            self._medium = medium
        elif large is not None:
            self._large = large

        if (
            self._string is None and self._small is None
            and self._medium is None and self._large is None
        ):
            raise Exception('Failed to initialize states.')

    @property
    def n(self) -> int:
        """Get number of states."""
        if self._n is not None:
            return self._n
        if self._small is not None:
            self._n = self._small.shape[0]
            return self.n
        if self._medium is not None:
            self._n = self._medium.shape[0]
            return self.n
        if self._large is not None:
            self._n = self._large.shape[0]
            return self.n
        if self._string is not None:
            self._n = self._string.shape[0]
            return self.n

    @property
    def string(self):
        """Get states as list of strings."""
        if self._string is not None:
            return self._string
        if self._small is not None:
            self._string = _small_to_string(self._small)
            return self.string
        if self._medium is not None:
            self._small = _medium_to_small(self._medium)
            return self.string
        if self._large is not None:
            self._small = _large_to_small(self._large)
            return self.string
        raise Exception('No state data.')

    @property
    def small(self):
        """Get states as array of shape (n, 2) of type uint32."""
        if self._small is not None:
            return self._small
        if self._medium is not None:
            self._small = _medium_to_small(self._medium)
            return self.small
        if self._large is not None:
            self._small = _large_to_small(self._large)
            return self.small
        if self._string is not None:
            self._small = _string_to_small(self._string)
            return self.small
        raise Exception('No state data.')

    @property
    def medium(self):
        """Get states as array of shape (n, 2, 8) of type uint8."""
        if self._medium is not None:
            return self._medium
        if self._small is not None:
            self._medium = _small_to_medium(self._small)
            return self.medium
        if self._large is not None:
            self._medium = _large_to_medium(self._large)
            return self.medium
        if self._string is not None:
            self._small = _string_to_small(self._string)
            return self.medium
        raise Exception('No state data.')

    @property
    def large(self):
        """Get states as array of shape (n, 4, 4) of type uint8."""
        if self._large is not None:
            return self._large
        if self._small is not None:
            self._large = _small_to_large(self._small)
            return self.large
        if self._medium is not None:
            self._large = _medium_to_large(self._medium)
            return self.large
        if self._string is not None:
            self._small = _string_to_small(self._string)
            return self.large
        raise Exception('No state data.')

    @property
    def valid(self):
        """Get valid states."""
        if self._valid is not None:
            return self._valid
        self._valid = States(large=self.large[self.validate])
        return self.valid

    @property
    def validate(self):
        """Get mask of states that pass validation."""
        if self._validate is not None:
            return self._validate
        self._validate = _validate(self.large)
        return self.validate

    @property
    def next(self):
        """Get next states."""
        if self._next is not None:
            return self._next
        nextStates = self.large + 0
        self._reward = _next(nextStates)
        self._next = States(large=nextStates)
        return self.next

    @property
    def reward(self):
        """Get rewards for advancing states."""
        if self._reward is not None:
            return self._reward
        self.next
        return self.reward

    @property
    def rotations(self):
        """Get rotations of states."""
        if self._rotations is not None:
            return self._rotations
        self._rotations = _rotations(self.large)
        return self.rotations

    @property
    def terminal(self):
        """Get mask of terminal states."""
        if self._terminal is not None:
            return self._terminal
        self._terminal = jnp.logical_not(_validate(self.rotations).any(axis=1))
        return self.terminal

    @property
    def live(self):
        """Get live states."""
        if self._live is not None:
            return self._live
        self._live = States(large=self.large[jnp.logical_not(self.terminal)])
        return self.live

    def rotate(self, actions):
        """Rotate states."""
        return States(large=_choose_rotation(self.rotations, actions))

    def add_tile(self, key):
        """Add new tile to board."""
        key1, key2 = random.split(key)
        rand1 = random.uniform(key1, (self.n,))
        rand2 = random.uniform(key2, (self.n,))
        nextStates = self.large + 0
        _add_tile(nextStates, rand1, rand2)
        return States(large=nextStates)


def _string_to_small(strings: List[State]):
    """Convert state string to (2,) array of type uint32."""
    return jnp.array(
        [[hex(string[:8]), hex(string[:7:-1])] for string in strings],
        jnp.uint32
    )


def _small_to_string(small):
    """Convert state (2,) array of type uint32 to string."""
    small = small.tolist()
    return [f'{top:0>8x}' + f'{bottom:0>8x}'[::-1] for top, bottom in small]


@jit
@vmap
def _small_to_medium(small):
    """Convert state (2,) array of type uint32 to (2,8) array of type uint8."""
    shifts = jnp.arange(0, 32, 4, jnp.uint32).reshape(1, 8).repeat(2, axis=0)
    padded = small.reshape(2, 1).repeat(8, axis=1)
    medium = ((padded << shifts) >> 28).astype(jnp.uint8)
    return medium


@jit
@vmap
def _small_to_large(small):
    """Convert state (2,) array of type uint32 to (4,4) array of type uint8."""
    shifts = jnp.array([[0, 4, 8, 12, 28, 24, 20, 16],
                        [28, 24, 20, 16, 0, 4, 8, 12]], jnp.uint32)
    padded = small.reshape(2, 1).repeat(8, axis=1)
    large = ((padded << shifts) >> 28).astype(jnp.uint8).reshape(4, 4)
    return large


@jit
@vmap
def _medium_to_small(medium):
    """Convert state (2,8) array of type uint8 to (2,) array of type uint32."""
    shifts = jnp.arange(28, -4, -4, jnp.uint32).reshape(1, 8).repeat(2, axis=0)
    small = (medium << shifts).sum(1)
    return small


@jit
def _medium_to_large(medium):
    """Convert (2,8) array of type uint8 to (2,4,4) array ot type uint8."""
    small = _medium_to_small(medium)
    large = _small_to_large(small)
    return large


@jit
@vmap
def _large_to_small(large):
    """Convert state (4,4) array of type uint8 to (2,) array of type uint32."""
    shifts = jnp.array([[28, 24, 20, 16, 0, 4, 8, 12],
                        [0, 4, 8, 12, 28, 24, 20, 16]], jnp.uint32)
    small = (large.reshape(2, 8) << shifts).sum(1)
    return small


@jit
def _large_to_medium(large):
    """Convert state (4,4) array of type uint8 to (2,8) array of type uint8."""
    small = _large_to_small(large)
    medium = _small_to_medium(small)
    return medium


@jit
@vmap
def _init_state(key):
    """Shuffle initial state arrays."""
    key1, key2 = random.split(key, 2)
    weights = jnp.array([0, 0.9, 0.1])
    tile1, tile2 = random.choice(key1, 3, (2,), p=weights)
    init = jnp.array([tile1, tile2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    permutation = random.permutation(key2, init).reshape(4, 4)
    return permutation.astype(jnp.uint8)


def numba_to_jax(dtype):
    """Cast result to jax array of type dtype."""
    def factory(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return jnp.array(result.copy_to_host(), dtype)
        return wrapper
    return factory


@numba_to_jax(jnp.bool_)
@guvectorize(
    ['void(u1[:,:],b1[:])'], '(n,n)->()', target=SETTINGS.target, nopython=True
)
def _validate(state, valid):
    """Validate state has next."""
    valid[0] = False
    for x in range(state.shape[0]):
        for y1 in range(state.shape[1] - 1):
            for y2 in range(y1 + 1, state.shape[1]):
                if state[x, y2] == 0:
                    continue
                elif state[x, y1] == 0:
                    valid[0] = True
                else:
                    if state[x, y1] == state[x, y2]:
                        valid[0] = True
                    break


@numba_to_jax(jnp.uint32)
@guvectorize(
    ['void(u1[:,:],u4[:])'], '(n,n)->()', target=SETTINGS.target, nopython=True
)
def _next(state, reward):
    """Advance to the next state."""
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


@jit
@vmap
def _rotations(state):
    """Calculate all rotations of state."""
    return jnp.stack([jnp.rot90(state, i) for i in range(4)])


@jit
def _choose_rotation(rotations, actions):
    """Choose rotation based on action."""
    return jnp.take_along_axis(rotations, actions.reshape(-1, 1, 1, 1), 1).\
        reshape(-1, 4, 4)


@numba_to_jax(jnp.bool_)
@guvectorize(
    ['void(u1[:,:], f4[:], f4[:], u1[:])'], '(n,n),(),()->()',
    target=SETTINGS.target, nopython=True
)
def _add_tile(state, rand1, rand2, success):
    """Add new tile to states."""
    num_zero = 0
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y] == 0:
                num_zero += 1
    if num_zero != 0:
        nth_zero = int(num_zero * rand1[0])
        x, y, count = 0, 0, 0
        for _x in range(state.shape[0]):
            for _y in range(state.shape[1]):
                if state[_x, _y] == 0:
                    if count == nth_zero:
                        x, y = _x, _y
                    count += 1
        state[x, y] = 1 if rand2[0] < 0.9 else 2
    success[0] = num_zero != 0
