"""Define FastAPI object."""
import jax
from fastapi import FastAPI
from jax import numpy as jnp
from jax import random
from pydantic import BaseModel, BaseSettings, conint
from typing import List

app = FastAPI()


class Settings(BaseSettings):
    """Pydantic model of app settings."""

    device: str = jax.default_backend()
    target: str = 'cuda' if jax.default_backend() == 'gpu' else 'cpu'


SETTINGS = Settings()

Positive = conint(ge=0)


class Random(BaseModel):
    """Pydantic model for jax random key."""

    count: Positive = 0
    seed: Positive

    @property
    def key(self):
        """Fetch key as jax array."""
        return jnp.array([self.count, self.seed], jnp.uint32)

    def split(self, num):
        """Split random key into n children keys."""
        children = random.split(self.key, num).copy().tolist()
        return [Random(count=child[0], seed=child[1]) for child in children]


@app.get('/rand/init/{seed}', response_model=Random)
def random_init(seed: Positive):
    """Initialize random key."""
    return Random(seed=seed)


class RandomSplit(BaseModel):
    """Pydantic model for response to split route."""

    parent: Random
    children: List[Random]


@app.put('/rand/split', response_model=RandomSplit)
def random_split(parent: Random, num: Positive):
    """Split random key into n children keys."""
    children = parent.split(num)
    return RandomSplit(parent=parent, children=children)
