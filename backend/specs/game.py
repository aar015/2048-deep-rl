"""deep2048 game specs."""
from .numeric import Unsigned
from ..engine.state import State
from pydantic import BaseModel
from typing import Optional


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
    can: GameCan
    q: Optional[GameQ]

    class Config:
        """Pydantic config."""

        allow_mutation = False
