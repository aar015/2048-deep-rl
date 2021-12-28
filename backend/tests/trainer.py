"""Test trainer code."""
import time
import warnings
from src.trainer import Trainer
from jax.random import PRNGKey, split
from numba.core.errors import NumbaPerformanceWarning
from src.network import NetworkSpec, Activation

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

key = PRNGKey(0)

key, subkey = split(key)
spec = NetworkSpec(**{
    'name': 'test_network',
    'sym_layers': [
        {'width': 16, 'activation': Activation.relu},
        {'width': 32, 'activation': Activation.relu},
        {'width': 16, 'activation': Activation.relu},
    ],
    'asym_layers': [
        {'width': 32, 'activation': Activation.relu},
        {'width': 16, 'activation': Activation.relu},
        {'width': 8, 'activation': Activation.relu},
        {'width': 4, 'activation': Activation.relu},
    ],
})
network = spec.init(subkey)

key, subkey = split(key)
trainer = Trainer(**{
    'num_epochs': int(1e1),
    'batch_play': int(1e2),
    'batch_learn': int(1e3),
})
start = time.time()
trained_network = trainer.train(subkey, network)
print(f'Time: {time.time() - start:.3f} s')

# print(trained_network)
