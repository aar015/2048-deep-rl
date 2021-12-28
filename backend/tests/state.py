"""Test state code."""
from src.state import States
from jax import random


def main():
    """Start tests."""
    n = 5
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)
    states = States(key1, n)
    print(states.string)
    print(states.next.string)
    print(states.next.add_tile(key2).string)
    print(states.string)
    print(states.medium.device())
    print(states.large.device())


if __name__ == '__main__':
    main()
