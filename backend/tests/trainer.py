"""Test trainer code."""
from src.trainer import Trainer
from jax.random import PRNGKey, split
from src.network import NetworkDispatch, Activation
import time

key = PRNGKey(0)

key, subkey = split(key)
dispatch = NetworkDispatch(**{
    'sym_layers': [
        {'width': 256, 'activation': Activation.relu},
        {'width': 128, 'activation': Activation.relu},
        {'width': 64, 'activation': Activation.relu},
    ],
    'asym_layers': [
        {'width': 64, 'activation': Activation.relu},
        {'width': 32, 'activation': Activation.relu},
        {'width': 16, 'activation': Activation.relu},
        {'width': 8, 'activation': Activation.relu},
    ]
})
params = dispatch.init_params(subkey)
network = params.network

key, subkey = split(key)
trainer = Trainer(**{
    'num_epochs': 10000,
})
start = time.time()
trained_network = trainer.train(subkey, network)
print(f'Time: {time.time() - start:.3f} s')
