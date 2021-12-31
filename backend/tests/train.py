"""Test trainer code."""
from jax.random import PRNGKey, split
from src.network import NetworkSpec, Activation
from src.random import Random
from src.train import Trainer
from time import time


def train():
    """Train Network."""
    key = PRNGKey(0)
    key, subkey = split(key)
    spec = NetworkSpec(**{
        'name': 'test_network',
        'seed': Random(seed=subkey[1], index=subkey[0]),
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
    network = spec.init()
    trainer = Trainer(**{
        'num_epochs': int(1e2),
        'batch_play': int(1e4),
        'batch_learn': int(1e5),
    })
    key, subkey = split(key)
    start = time()
    trainer.train(subkey, network, 'Trained Network')
    print(f'Time: {time() - start:.3f} s')


if __name__ == '__main__':
    train()
