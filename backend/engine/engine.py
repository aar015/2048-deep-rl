"""deep2048 game engine."""
from jax import numpy as jnp
from jax.random import split
from .batch import Batch


class Engine(object):
    """Engine to play 2048."""

    def __init__(self, key, n=1, batch=None, scores=None):
        """Initialize 2048 game engine."""
        self._key = key
        if batch is None:
            self._key, subkey = split(self._key)
            self._n = n
            self._batch = Batch(key=subkey, n=self._n)
        else:
            self._n = batch.n
            self._batch = batch
        if scores is None:
            self._scores = jnp.zeros(self._n, jnp.uint32)
        elif scores.shape[0] != self._n:
            raise Exception('Score size must match batch size.')
        else:
            self._scores = jnp.array(scores, jnp.uint32)

    @property
    def key(self):
        """Get current engine key."""
        return self._key

    @property
    def n(self):
        """Get current engine size."""
        return self._n

    @property
    def batch(self):
        """Get current engine batch."""
        return self._batch

    @property
    def scores(self):
        """Get current engine scores."""
        return self._scores

    def next(self, actions=None, choose_actions=None):
        """Advance engine to next step using actions or choose_actions."""
        self._key, subkey1, subkey2 = split(self.key, 3)
        oldBatch = self.batch
        if actions is None:
            actions = choose_actions(subkey1, self.batch)
        rotated = oldBatch.rotate(actions)
        preBatch = rotated.next.rotate(-1 * actions)
        self._scores += rotated.reward
        self._batch = preBatch.add_tile(subkey2)
        return oldBatch, preBatch, self.batch, rotated.reward

    def run(self, choose_actions):
        """Advance engine until all states terminal using choose_actions."""
        while not jnp.all(self.batch.terminal):
            self._key, subkey1, subkey2 = split(self.key, 3)
            actions = choose_actions(subkey1, self.batch)
            rotated = self.batch.rotate(actions)
            self._scores += rotated.reward
            self._batch = rotated.next.rotate(-1 * actions).add_tile(subkey2)
        return self.batch, self.scores
