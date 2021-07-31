"""Test agent code."""
from jax import numpy as jnp
from jax.random import PRNGKey, split, uniform
from src.network import NetworkDispatch, Activation
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


def play(key, params, n):
    """Play Game."""
    key, subkey = split(key)
    states = States(subkey, n)
    score = jnp.zeros((n,))
    turns = []
    while not states.terminal.all():
        actions = choose_action(params, states)
        rotated = states.rotate(actions)
        score += rotated.reward
        states = rotated.next.add_tile(key)
        turns.append({'state': rotated, 'score': score})
    for turn in turns:
        turn['future-reward'] = score - turn['score']
    states = States(
        medium=jnp.concatenate([turn['state'].medium for turn in turns])
    )
    future_rewards = jnp.concatenate([turn['future-reward'] for turn in turns])
    mask = jnp.logical_not(states.terminal)
    return states.valid, future_rewards[mask]


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
states, future_rewards = play(key, params, 10000)
print(f'Time: {time.time() - start:.3f} s')
print(states.n)
# print(states.string)
# print(future_rewards)
