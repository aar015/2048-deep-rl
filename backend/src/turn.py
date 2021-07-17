"""Define turn in 2048 game."""
from .app import app, Positive, Random
from .state import State, States
from enum import IntEnum
from pydantic import BaseModel
from typing import Optional
from jax import numpy as jnp


def execute(key, states: States, actions):
    """Execute a turn."""
    rotated = states.rotate(actions)
    return rotated.reward, rotated.next


class Action(IntEnum):
    """Action enum for typing."""

    left = 0
    up = 1
    right = 2
    down = 3


class TurnTicket(BaseModel):
    """Ticket to execute turn."""

    state: State
    action: Action
    random: Random


class Turn(TurnTicket):
    """Symmetrize function output."""

    success: bool
    reward: Optional[Positive]
    nextState: Optional[State]
    terminal: Optional[bool]


@app.post('/game/execute', response_model=Turn)
def execute_turn(ticket: TurnTicket):
    """Execute a turn."""
    state = States(string=[ticket.state])
    if not state.validate[0]:
        return Turn(**ticket.dict(), success=False)
    action = jnp.array([ticket.action], jnp.int8)
    key = ticket.random.key
    reward, nextState = execute(key, state, action)
    reward = reward.tolist()[0]
    nextState = nextState.rotate(-1 * action)
    terminal = nextState.terminal
    return Turn(
        **ticket.dict(), success=True, reward=reward,
        nextState=nextState.string, terminal=terminal
    )
