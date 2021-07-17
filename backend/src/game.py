"""Implement routes to play game."""
from .app import app
from .random import Positive, Random
from .state import State, States
from enum import IntEnum
from pydantic import BaseModel
from jax import numpy as jnp


class Action(IntEnum):
    """Action enum for typing."""

    left = 0
    up = 1
    right = 2
    down = 3


class Turn(BaseModel):
    """Symmetrize function output."""

    state: State
    action: Action
    reward: Positive
    nextState: State
    terminal: bool


@app.get('/game/init', response_model=State)
def initialize_board(seed: Positive, index: Positive = 0):
    """Generate random initial board state."""
    key = Random(seed=seed, index=index)
    states = States(key.key, 1)
    return states.string[0]


@app.get('/game/execute', response_model=Turn)
def execute_action(
    state: State, action: Action, seed: Positive, index: Positive = 0
):
    """Execute an action on the board state."""
    states = States(string=[state])
    if not states.validate[0]:
        terminal = states.terminal.tolist()[0]
        return Turn(
            state=state, action=action, reward=0,
            nextState=state, terminal=terminal
        )
    key = Random(seed=seed, index=index).key
    actions = jnp.array([action], jnp.int8)
    rotated = states.rotate(actions)
    reward = rotated.reward.tolist()[0]
    nextStates = rotated.next.add_tile(key)
    nextState = nextStates.rotate(-1 * action).string[0]
    terminal = nextStates.terminal.tolist()[0]
    return Turn(
        state=state, action=action, reward=reward,
        nextState=nextState, terminal=terminal
    )
