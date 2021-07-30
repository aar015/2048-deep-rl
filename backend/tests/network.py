"""Test network code."""
from jax.random import PRNGKey, split
from src.activation import Activation
from src.network import NetworkDispatch
from src.state import States

dispatch = NetworkDispatch(**{
    'sym_layers': [
        {'width': 8, 'activation': Activation.relu}
    ],
    'asym_layers': [
        {'width': 8, 'activation': Activation.relu}
    ]
})

key = PRNGKey(0)

key, subkey = split(key)
params = dispatch.init_params(subkey)

state = '0123456789abcdef'
states = States(string=[state, state[::-1], state[1::2] + state[::2],
                state[::2][::-1] + state[1::2][::-1]])
print(params.network)
# print(params.predict(states))
