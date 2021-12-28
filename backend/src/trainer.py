"""Implement routes to query agent."""
from typing import Optional
from .network import Network, NetworkParams, Positive, _forward
from .state import States, _validate
from functools import partial
from jax import grad, jit, tree_multimap
from jax import numpy as jnp
from jax.random import split, uniform, choice
from pydantic import BaseModel, confloat

Fraction = confloat(ge=0, le=1)


class Trainer(BaseModel):
    """Pydantic model for network trainer."""

    num_epochs: Positive = int(1e2)
    batch_play: Positive = int(1e3)
    batch_learn: Positive = int(1e4)
    exploration: Fraction = 1e-3
    exploration_decay: Fraction = 0
    learning_rate: float = 1e-3
    learning_rate_decay: Fraction = 0

    def train(self, key, network: Network, new_name: Optional[str] = None):
        """Train network."""
        params = network.params
        for epoch in range(self.num_epochs):
            print(f'Epoch: {epoch}')
            key, subkey1, subkey2 = split(key, 3)
            states, future_rewards = self._play(subkey1, params)
            params = self._update(subkey2, params, states, future_rewards)
        return params.construct(network.name if new_name is None else new_name)

    def _play(self, key, params):
        key, subkey = split(key)
        states = States(subkey, self.batch_play)
        score = jnp.zeros((self.batch_play,))
        turns = []
        while not states.terminal.all():
            key, subkey1, subkey2 = split(key, 3)
            actions = self._choose_action(subkey1, params, states)
            rotated = states.rotate(actions)
            score += rotated.reward
            states = rotated.next.add_tile(subkey2)
            turns.append({'state': rotated, 'score': score})
        for turn in turns:
            turn['future-reward'] = score - turn['score']
        states = States(
            medium=jnp.concatenate([turn['state'].medium for turn in turns])
        )
        future_rewards = jnp.concatenate(
            [turn['future-reward'] for turn in turns]
        )
        max_tile = 2 ** states.medium.max()
        high_score = int(score.max())
        print(
            f'Max Tile: {max_tile}, High Score {high_score}'
        )
        mask = jnp.logical_not(states.terminal)
        return states.valid, future_rewards[mask]

    def _choose_action(self, key, params, states):
        valid_actions = _validate(states.rotations)
        predictions = params.predict(states)
        valid_predictions = valid_actions * predictions
        actions = valid_predictions.argmax(1)
        key1, key2 = split(key)
        rands = uniform(key1, (states.n,))
        random_predictions = uniform(key2, (states.n, 4)) * valid_predictions
        random_actions = random_predictions.argmax(1)
        actions = jnp.where(rands < self.exploration, random_actions, actions)
        return actions

    def _update(self, key, params, states, future_rewards):
        key, subkey = split(key)
        indices = choice(subkey, states.n, (self.batch_learn,), False)
        x = states.medium[indices]
        y = future_rewards[indices]
        params = params.dict()
        [sym_layers, asym_layers] = _update(
            params['sym_layers'], params['asym_layers'],
            tuple(params['activations']), self.learning_rate, x, y
        )
        return NetworkParams(**{
            'sym_layers': sym_layers,
            'asym_layers': asym_layers,
            'activations': params['activations']
        })


def _loss(sym_layers, asym_layers, activations, x, y):
    y_hat = _forward(sym_layers, asym_layers, activations, x)
    return jnp.mean((y_hat - y) ** 2)


@partial(jit, static_argnums=[2])
def _update(sym_layers, asym_layers, activations, learning_rate, x, y):
    sym_grads, asym_grads = grad(_loss, argnums=[0, 1])(
        sym_layers, asym_layers, activations, x, y
    )
    return [tree_multimap(
        lambda p, g: p - learning_rate * g, layer, grads
    ) for layer, grads in ((sym_layers, sym_grads), (asym_layers, asym_grads))]
