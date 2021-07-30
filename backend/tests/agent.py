"""Test agent code."""
from jax import numpy as jnp
from jax.random import PRNGKey, split, uniform
from src.activation import Activation
from src.network import NetworkDispatch
from src.state import States, _validate
import time


def choose_action(params, states, key=None, exploration=0):
    """Choose action."""
    valid_actions = _validate(states.rotations)
    predictions = params.predict(states)
    valid_predictions = valid_actions * predictions
    actions = valid_predictions.argmax(1)
    if key is not None:
        key1, key2 = split(key)
        rands = uniform(key1, (states.n,))
        random_predictions = uniform(key2, (states.n, 4)) * valid_predictions
        random_actions = random_predictions.argmax(1)
        actions = jnp.where(rands < exploration, random_actions, actions)
    return actions


def execute_action(key, states, actions):
    """Execute action."""
    rotated = states.rotate(actions)
    nextStates = rotated.next.add_tile(key).rotate(-1 * actions)
    return nextStates


def play_game(key, params):
    """Play Game."""
    key, subkey = split(key)
    states = States(subkey, 1000)
    while not states.terminal.all():
        # print(states.live.string)
        actions = choose_action(params, states)
        # print(actions)
        states = execute_action(key, states, actions)


key = PRNGKey(0)

key, subkey = split(key)
dispatch = NetworkDispatch(**{
    'sym_layers': [
        {'width': 8, 'activation': Activation.relu}
    ],
    'asym_layers': [
        {'width': 8, 'activation': Activation.relu}
    ]
})
params = dispatch.init_params(subkey)

key, subkey = split(key)
start = time.time()
play_game(key, params)
print(f'Time: {time.time() - start:.3f} s')
