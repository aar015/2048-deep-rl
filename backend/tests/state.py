"""Test state code."""
from src.state import States
from jax import random
from time import time


def main():
    """Start tests."""
    n = int(1e6)
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    start = time()
    states = States(key1, n)
    strings = states.string
    elapsed = time() - start
    print(f'First Time: {elapsed*1e3:.2f} ms')
    print(strings[:5])
    start = time()
    states = States(key2, n)
    strings = states.string
    elapsed = time() - start
    print(f'Second Time: {elapsed*1e3:.2f} ms')
    print(strings[:5])


if __name__ == '__main__':
    main()
