"""Test network code."""
from jax.random import PRNGKey, split
from src.network import Activation, NetworkSpec
from src.state import States

dispatch = NetworkSpec(**{
    'name': 'test1',
    'sym_layers': [
        {'width': 8, 'activation': Activation.relu}
    ],
    'asym_layers': [
        {'width': 8, 'activation': Activation.relu}
    ]
})

key = PRNGKey(0)

key, subkey = split(key)
network = dispatch.init(subkey)

state = '0123456789abcdef'
states = States(string=[state, state[::-1], state[1::2] + state[::2],
                state[::2][::-1] + state[1::2][::-1]])
print(network)
print(network.params.dict())
print(network.predict(states))
