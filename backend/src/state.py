"""Define state of 2048 game."""
from .app import app
from functools import partial
from jax import jit, vmap
from jax import numpy as jnp
from typing import List
from pydantic.types import constr

StateString = constr(
    strip_whitespace=True, to_lower=True,
    min_length=16, max_length=16, regex='^[a-f0-9]*$'
)

hex = partial(int, base=16)


def string_to_small(strings: List[StateString]):
    """Convert state string to (2,) array of type uint32.

    !!!----------Warning----------!!!
    This function is slow.
    !!!---------------------------!!!
    """
    return jnp.array(
        [[hex(string[:8]), hex(string[:7:-1])] for string in strings],
        jnp.uint32
    )


@jit
@vmap
@jit
def small_to_large(small):
    """Convert state (2,) array of type uint32 to (4,4) array of type uint8."""
    shifts = jnp.array([[0, 4, 8, 12, 28, 24, 20, 16],
                        [28, 24, 20, 16, 0, 4, 8, 12]], jnp.uint32)
    padded = small.reshape(2, 1).repeat(8, axis=1)
    expanded = ((padded << shifts) >> 28).astype(jnp.uint8).reshape(4, 4)
    return expanded


@jit
@vmap
@jit
def large_to_small(large):
    """Convert state (4,4) array of type uint8 to (2,) array of type uint32."""
    shifts = jnp.array([[28, 24, 20, 16, 0, 4, 8, 12],
                        [0, 4, 8, 12, 28, 24, 20, 16]], jnp.uint32)
    compressed = (large.reshape(2, 8) << shifts).sum(1)
    return compressed


def small_to_string(small):
    """Convert state (2,) array of type uint32 to string."""
    small = small.copy().tolist()
    return [f'{top:0>8x}' + f'{bottom:0>8x}'[::-1] for top, bottom in small]


@app.get('/state/{string}')
def get_state(string: StateString):
    """Convert board string to list of tiles."""
    small = string_to_small([string])
    large = small_to_large(small)
    small = large_to_small(large)
    string = small_to_string(small)
    return string[0]
