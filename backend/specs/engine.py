"""deep2048 engine specs."""
from jax import numpy as jnp
import numpy as np
from pydantic import BaseModel
from typing import Optional, List
from .game import GameAction, GameCan, GameState
from .numeric import Positive
from .random import Key


class EngineInit(BaseModel):
    """Request to initialize 2048 game engine."""

    key: Key
    n: Positive = 1
    agent: Optional[str]

    class Config:
        """Pydantic config."""

        allow_mutation = False


class EngineState(EngineInit):
    """State of 2048 game engine."""

    games: List[GameState]

    @classmethod
    def from_engine(cls, engine, agent):
        """Construct engine state from jax arrays."""
        key = Key.from_array(engine.key)
        n = engine.n
        states = engine.batch.string
        can_move = np.array(jnp.logical_not(engine.batch.terminal))
        can_left = np.array(engine.batch.valid_mask)
        can_up = np.array(engine.batch.rotate(1).valid_mask)
        can_right = np.array(engine.batch.rotate(2).valid_mask)
        can_down = np.array(engine.batch.rotate(3).valid_mask)
        cans = [
            GameCan(move=move, left=left, up=up, right=right, down=down)
            for move, left, up, right, down
            in zip(can_move, can_left, can_up, can_right, can_down)
        ]
        games = [
            GameState(state=state, score=score, can=can)
            for state, score, can
            in zip(states, engine.scores, cans)
        ]
        return cls(key=key, n=n, agent=agent, games=games)


class EngineAction(EngineInit):
    """Request to perform actions in 2048 game engine."""

    games: List[GameAction]

    @property
    def state(self):
        """Get corresponding EngineState object."""
        return EngineState(**self.dict())
