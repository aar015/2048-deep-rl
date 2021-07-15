"""Define turn in 2048 game."""
from .app import app, Positive, Random
from .state import StateString
from enum import IntEnum
from pydantic import BaseModel
from typing import Optional, List


class Action(IntEnum):
    """Action enum for typing."""

    left = 0
    up = 1
    right = 2
    down = 3


class TurnTicket(BaseModel):
    """Symmetrize function input."""

    state: List[StateString]
    action: List[Action]
    random: Random


class TurnSummary(BaseModel):
    """Symmetrize function output."""

    state: StateString
    action: Action
    reward: Positive
    nextState: StateString
    terminal: bool
    score: Positive
    futureScore: Optional[Positive] = None


@app.post('/turn', response_model=TurnSummary)
def submit(ticket: TurnTicket):
    """Symmetrize input turn."""
    pass
