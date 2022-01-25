"""deep2048 game actions."""
from functools import wraps
from jax import jit, vmap, random
from jax import numpy as jnp
from numba import guvectorize
from numba.core.errors import NumbaDeprecationWarning, NumbaPerformanceWarning
from numpy import ndarray
from warnings import simplefilter
from ..settings import SETTINGS

simplefilter('ignore', category=NumbaDeprecationWarning)
simplefilter('ignore', category=NumbaPerformanceWarning)


@jit
@vmap
def init(key):
    """Initialize game state."""
    key1, key2 = random.split(key, 2)
    tile1, tile2 = random.choice(key1, 3, (2,), p=jnp.array([0, 0.9, 0.1]))
    return random.permutation(
        key2,
        jnp.array([tile1, tile2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ).reshape(4, 4).astype(jnp.uint8)


@jit
@vmap
def rotations(state):
    """Get all rotations of state."""
    return jnp.stack([jnp.rot90(state, i) for i in range(4)])


@jit
def choose_rotation(rotations, actions):
    """Choose rotation based on action."""
    return jnp.take_along_axis(rotations, actions.reshape(-1, 1, 1, 1), 1).\
        reshape(-1, 4, 4)


def numba_to_jax(dtype):
    """Cast numba array to jax array of type dtype."""
    def factory(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if type(result) == ndarray:
                return jnp.array(result, dtype)
            return jnp.array(result.copy_to_host(), dtype)
        return wrapper
    return factory


@numba_to_jax(jnp.bool_)
@guvectorize(
    ['void(u1[:,:],b1[:])'],
    '(n,n)->()',
    target=SETTINGS.target,
    nopython=True
)
def validate(state, valid):
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
    ['void(u1[:,:],u4[:])'],
    '(n,n)->()',
    target=SETTINGS.target,
    nopython=True
)
def next(state, reward):
    """Advance to the next pre-state."""
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


@numba_to_jax(jnp.bool_)
@guvectorize(
    ['void(u1[:,:], f4[:], f4[:], u1[:])'],
    '(n,n),(),()->()',
    target=SETTINGS.target,
    nopython=True
)
def add_tile(state, rand1, rand2, success):
    """Add new tile to pre-state."""
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
