"""Define state of 2048 game."""
from .app import app, Random, Positive
from functools import partial
from jax import jit, vmap, random
from jax import numpy as jnp
from typing import List
from pydantic.types import constr


StateString = constr(
    strip_whitespace=True, to_lower=True,
    min_length=16, max_length=16, regex='^[a-f0-9]*$'
)

hex = partial(int, base=16)


def _string_to_small(strings: List[StateString]):
    """Convert state string to (2,) array of type uint32."""
    return jnp.array(
        [[hex(string[:8]), hex(string[:7:-1])] for string in strings],
        jnp.uint32
    )


@jit
@vmap
@jit
def _small_to_large(small):
    """Convert state (2,) array of type uint32 to (4,4) array of type uint8."""
    shifts = jnp.array([[0, 4, 8, 12, 28, 24, 20, 16],
                        [28, 24, 20, 16, 0, 4, 8, 12]], jnp.uint32)
    padded = small.reshape(2, 1).repeat(8, axis=1)
    expanded = ((padded << shifts) >> 28).astype(jnp.uint8).reshape(4, 4)
    return expanded


@jit
@vmap
@jit
def _large_to_small(large):
    """Convert state (4,4) array of type uint8 to (2,) array of type uint32."""
    shifts = jnp.array([[28, 24, 20, 16, 0, 4, 8, 12],
                        [0, 4, 8, 12, 28, 24, 20, 16]], jnp.uint32)
    compressed = (large.reshape(2, 8) << shifts).sum(1)
    return compressed


def _small_to_string(small):
    """Convert state (2,) array of type uint32 to string."""
    small = small.tolist()
    return [f'{top:0>8x}' + f'{bottom:0>8x}'[::-1] for top, bottom in small]


@jit
@vmap
@jit
def _init_state(key):
    """Shuffle initial state arrays."""
    key1, key2 = random.split(key, 2)
    weights = jnp.array([0, 0.9, 0.1])
    tile1, tile2 = random.choice(key1, 3, (2,), p=weights)
    init = jnp.array([tile1, tile2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    permutation = random.permutation(key, init).reshape(4, 4)
    return permutation


class States(object):
    """Batch of 2048 board states."""

    def __init__(self, key=None, n=100, string=None, small=None, large=None):
        """Initialize states."""
        self._string = None
        self._small = None
        self._large = None
        if key is None:
            if string is not None:
                self._string = string
            elif small is not None:
                self._small = small
            elif large is not None:
                self._large = large
        else:
            keys = random.split(key, n)
            self._large = _init_state(keys)
        if (
            self._string is None and self._small is None
            and self._large is None
        ):
            raise Exception('Must pass arguement to Stat init.')

    @property
    def string(self):
        """Get states as list of strings."""
        if self._string is not None:
            return self._string
        if self._small is not None:
            self._string = _small_to_string(self._small)
            return self.string
        if self._large is not None:
            self._small = _large_to_small(self._large)
            return self.string
        raise Exception('No state data.')


@app.put('/state/init', response_model=List[StateString])
def init(key: Random, n: Positive = 10):
    """Generate n random initial states."""
    states = States(key.key, n)
    return states.string
