"""deep2048 random specs."""
from .numeric import Unsigned
from jax import numpy as jnp
from jax import random
from pydantic import BaseModel, conint


class Key(BaseModel):
    """PRNG random key."""

    seed: Unsigned
    index: Unsigned = 0

    class Config:
        """Pydantic config."""

        allow_mutation = False

    @property
    def array(self):
        """Return random key as jax ndarray."""
        return jnp.array([self.index, self.seed], jnp.uint32)

    def split(self, num: int = 2):
        """Split random key into n children keys."""
        return [Key(seed=child[1], index=child[0])
                for child in random.split(self.array, num)]


class SplitKey(BaseModel):
    """Request to split PRNG random key."""

    key: Key
    n: conint(ge=2)

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def split(self):
        """Split random key."""
        return self.key.split(self.n)
