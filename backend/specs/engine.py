"""deep2048 engine specs."""
import numpy as np
from .game import GameCan, GameState
from .numeric import Positive
from .random import Key
from ..engine.batch import Batch
from jax import numpy as jnp
from pydantic import BaseModel
from typing import Optional, List


class EngineInit(BaseModel):
    """Request to initialize 2048 engine."""

    key: Key
    n: Positive = 1
    agent: Optional[str]

    class Config:
        """Pydantic config."""

        allow_mutation = False


class EngineState(EngineInit):
    """State of 2048 engine."""

    games: List[GameState]

    class Config:
        """Pydantic config."""

        allow_mutation = False

    @classmethod
    def from_batch(cls, key: Key, batch: Batch, scores, agent):
        """Construct engine state from batch."""
        n = batch.n
        states = batch.string
        can_move = np.array(jnp.logical_not(batch.terminal))
        can_left = np.array(batch.validate)
        can_up = np.array(batch.rotate(1).validate)
        can_right = np.array(batch.rotate(2).validate)
        can_down = np.array(batch.rotate(3).validate)
        cans = [
            GameCan(move=move, left=left, up=up, right=right, down=down)
            for move, left, up, right, down
            in zip(can_move, can_left, can_up, can_right, can_down)
        ]
        games = [
            GameState(state=state, score=score, can=can)
            for state, score, can
            in zip(states, scores, cans)
        ]
        return cls(key=key, n=n, agent=agent, games=games)
