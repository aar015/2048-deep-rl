"""Implement routes to split random keys."""
from .app import app
from .numeric import Unsigned
from jax import numpy as jnp
from jax import random
from pydantic import BaseModel, conint
from typing import List


class Random(BaseModel):
    """Pydantic model for jax random key."""

    seed: Unsigned
    index: Unsigned = 0

    class Config:
        """Pydantic config."""

        allow_mutation = False

    @property
    def key(self):
        """Return random key as jax ndarray."""
        return jnp.array([self.index, self.seed], jnp.uint32)

    def split(self, num: int = 2):
        """Split random key into n children keys."""
        return [Random(seed=child[1], index=child[0])
                for child in random.split(self.key, num)]


class SplitSpec(BaseModel):
    """Request to split random key."""

    key: Random
    num: conint(ge=2)

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def split(self):
        """Split random key."""
        return self.key.split(self.num)


@app.get('/random/split', response_model=List[Random])
def random_split(spec: SplitSpec):
    """Split random key into n children keys."""
    return spec.split()
