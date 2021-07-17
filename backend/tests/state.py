"""Test state code."""
from src.state import States
from jax import random
from jax import numpy as jnp


def main():
    """Start tests."""
    n = 5
    key = random.PRNGKey(0)
    states = States(key, n)
    print(jnp.argmax(states.large[0]))


if __name__ == '__main__':
    main()
