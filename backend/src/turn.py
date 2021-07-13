"""Define turn in 2048 game."""
from pydantic import BaseModel
from ..api import api


class Turn(BaseModel):
    """Symmetrize function input."""

    state: str
    action: int


class MoveResponse(BaseModel):
    """Symmetrize function output."""

    state: str
    reward: int
    terminal: bool


@api.post('/environ/turn', response_model=MoveResponse)
async def move(board: Turn):
    """Symmetrize input turn."""
    return MoveResponse(state='',  reward=0, terminal=True)