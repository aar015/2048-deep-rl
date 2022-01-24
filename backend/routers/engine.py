"""deep2048 engine router."""
from ..engine.batch import Batch
from ..specs.engine import EngineInit, EngineState
from fastapi import APIRouter
from jax import numpy as jnp

router = APIRouter()


@router.put('/init', response_model=EngineState)
def state_init(spec: EngineInit):
    """Initialize board state."""
    key1, key2 = spec.key.split()
    batch = Batch(key=key1.array, n=spec.n)
    scores = jnp.zeros(spec.n)
    return EngineState.from_batch(key2, batch, scores, spec.agent)
