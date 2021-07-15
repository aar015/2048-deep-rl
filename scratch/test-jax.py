"""Test jax."""
import jax
import jax.numpy as jnp
from functools import partial
from numba import guvectorize

if jax.default_backend() == 'gpu':
    TARGET = 'cuda'
else:
    TARGET = 'cpu'

hex = partial(int, base=16)


@jax.vmap
@jax.jit
def compress(expanded):
    """Compress state (4, 4) array to state (2, 1) array."""
    shifts = jnp.array([[28, 24, 20, 16, 0, 4, 8, 12],
                        [0, 4, 8, 12, 28, 24, 20, 16]], jnp.uint32)
    compressed = (expanded.reshape(2, 8) << shifts).sum(1)
    return compressed


@jax.vmap
@jax.jit
def expand(compressed):
    """Expand state (2,) array to state (4, 4) array."""
    shifts = jnp.array([[0, 4, 8, 12, 28, 24, 20, 16],
                        [28, 24, 20, 16, 0, 4, 8, 12]], jnp.uint32)
    padded = compressed.reshape(2, 1).repeat(8, axis=1)
    expanded = ((padded << shifts) >> 28).astype(jnp.uint8).reshape(4, 4)
    return expanded


@guvectorize(
    ['void(u1[:,:],b1[:])'], '(n,n)->()', target=TARGET, nopython=True
)
def validate(state, can):
    """Validate turn is advancable."""
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


@guvectorize(
    ['void(u1[:,:],u4[:])'], '(n,n)->()', target=TARGET,
    nopython=True
)
def advance(state, reward):
    """Advance the turn."""
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


def rotations(states):
    """Find every rotation of states."""
    rot90 = jnp.rot90(states, 1, (-2, -1))
    rot180 = jnp.rot90(states, 2, (-2, -1))
    rot270 = jnp.rot90(states, 3, (-2, -1))
    ro


def main():
    """Execute jax test."""
    string = '0123456789abcdef'
    compressed = jnp.array([hex(string[:8]), hex(string[:7:-1])], jnp.uint32).\
        reshape((1, 2)).repeat(int(1e1), axis=0)
    expanded = expand(compressed)
    action = jnp.arange(10)
    print(jnp.rot90(expanded, action, axes=(-2, -1)))


if __name__ == '__main__':
    main()
