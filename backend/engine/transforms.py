"""deep2048 game state size transforms."""
from functools import partial
from jax import jit, vmap
from jax import numpy as jnp

hex = partial(int, base=16)


def string_to_small(strings):
    """Convert state string to (2,) array of type uint32."""
    return jnp.array(
        [[hex(string[:8]), hex(string[:7:-1])] for string in strings],
        jnp.uint32)


def small_to_string(small):
    """Convert state (2,) array of type uint32 to string."""
    small = small.tolist()
    return [f'{top:0>8x}' + f'{bottom:0>8x}'[::-1] for top, bottom in small]


@jit
@vmap
def small_to_medium(small):
    """Convert state (2,) array of type uint32 to (2,8) array of type uint8."""
    shifts = jnp.arange(0, 32, 4, jnp.uint32).reshape(1, 8).repeat(2, axis=0)
    padded = small.reshape(2, 1).repeat(8, axis=1)
    return ((padded << shifts) >> 28).astype(jnp.uint8)


@jit
@vmap
def small_to_large(small):
    """Convert state (2,) array of type uint32 to (4,4) array of type uint8."""
    shifts = jnp.array([
        [0, 4, 8, 12, 28, 24, 20, 16],
        [28, 24, 20, 16, 0, 4, 8, 12]],
        jnp.uint32)
    padded = small.reshape(2, 1).repeat(8, axis=1)
    return ((padded << shifts) >> 28).reshape(4, 4).astype(jnp.uint8)


@jit
@vmap
def medium_to_small(medium):
    """Convert state (2,8) array of type uint8 to (2,) array of type uint32."""
    shifts = jnp.arange(28, -4, -4, jnp.uint32).reshape(1, 8).repeat(2, axis=0)
    return (medium << shifts).sum(1)


@jit
@vmap
def large_to_small(large):
    """Convert state (4,4) array of type uint8 to (2,) array of type uint32."""
    shifts = jnp.array([
        [28, 24, 20, 16, 0, 4, 8, 12],
        [0, 4, 8, 12, 28, 24, 20, 16]],
        jnp.uint32)
    return (large.reshape(2, 8) << shifts).sum(1)
