"""deep2048 game specs."""
from enum import IntEnum
from pydantic import BaseModel
from typing import Optional
from .numeric import Unsigned
from .state import State


class GameCan(BaseModel):
    """Permitted game actions."""

    move: bool
    left: bool
    up: bool
    right: bool
    down: bool

    class Config:
        """Pydantic config."""

        allow_mutation = False


class GameQ(BaseModel):
    """Q-value for each game action."""

    left: Unsigned
    up: Unsigned
    right: Unsigned
    down: Unsigned

    class Config:
        """Pydantic config."""

        allow_mutation = False


class GameState(BaseModel):
    """State of 2048 game."""

    state: State
    score: Unsigned
    can: Optional[GameCan]
    q: Optional[GameQ]

    class Config:
        """Pydantic config."""

        allow_mutation = False


class Action(IntEnum):
    """Action in 2048 game engine."""

    left = 0
    up = 1
    right = 2
    down = 3


class GameAction(GameState):
    """Request to perform action on 2048 game state."""

    action: Optional[Action]
