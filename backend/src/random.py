"""Implement routes to split random keys."""
from .app import app
from jax import numpy as jnp
from jax import random
from pydantic import BaseModel, conint
from typing import List

Unsigned = conint(ge=0)


class Random(BaseModel):
    """Pydantic model for jax random key."""

    __slots__ = ('key')
    seed: Unsigned
    index: Unsigned = 0

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def __init__(self, **kwargs):
        """Initialize random key model."""
        super().__init__(**kwargs)
        key = jnp.array([self.index, self.seed], jnp.uint32)
        object.__setattr__(self, 'key', key)

    def split(self, num):
        """Split random key into n children keys."""
        children = random.split(self.key, num).copy().tolist()
        return [Random(seed=child[1], index=child[0]) for child in children]


@app.get('/random', response_model=List[Random])
def split_random_key(seed: Unsigned, index: Unsigned = 0, n: conint(ge=2) = 2):
    """Split random key into n children keys."""
    return Random(seed=seed, index=index).split(n)
