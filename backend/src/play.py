"""Implement routes to play game according to network."""
from .app import app
from .network import Network
from .numeric import Fraction, Positive, Unsigned
from .random import Random
from .state import State, States
from enum import IntEnum
from pydantic import BaseModel
from typing import List


class Action(IntEnum):
    """Action enum for typing."""

    left = 0
    up = 1
    right = 2
    down = 3


class TurnSpec(BaseModel):
    """Request to play a turn."""

    key: Random
    state: State
    action: Action

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def play(self):
        """Play turn."""
        states = States(string=[self.state])
        rotated = states.rotate(self.action)
        if not rotated.validate[0]:
            terminal = rotated.terminal.tolist()[0]
            return Turn(
                state=self.state, action=self.action, reward=0,
                nextState=self.state, terminal=terminal
            )
        reward = rotated.reward.tolist()[0]
        key = self.key.key
        nextStates = rotated.next.add_tile(key)
        nextState = nextStates.rotate(-1 * self.action).string[0]
        terminal = nextStates.terminal.tolist()[0]
        return Turn(
            state=self.state, action=self.action, reward=reward,
            nextState=nextState, terminal=terminal
        )


class Turn(BaseModel):
    """One turn in a game of 2048."""

    state: State
    action: Action
    reward: Unsigned
    nextState: State
    terminal: bool

    class Config:
        """Pydantic config."""

        allow_mutation = False


@app.get('/play/turn', response_model=Turn)
def play_turn(spec: TurnSpec):
    """Play turn of 2048."""
    return spec.play()


class GameSpec(BaseModel):
    """Request to play game according to network."""

    key: Random
    exploration: Fraction
    network: Network

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def play(self):
        """Play game."""
        pass


class Game(BaseModel):
    """One game of 2048."""

    max_score: Unsigned
    max_tile: Positive
    num_turns: Positive
    turns: List[Turn]

    class Config:
        """Pydantic config."""

        allow_mutation = False


@app.get('/play/game', response_model=Game)
def play_game(spec: GameSpec):
    """Play game of 2048."""
    return spec.play()


class BatchSpec(BaseModel):
    """Request to play batch of games according to network."""

    key: Random
    batch_size: Positive
    exploration_rate: Fraction
    newtwork: Network

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def play(self):
        """Play batch."""
        pass


class Batch(BaseModel):
    """Batch of games of 2048."""

    batch_size: Positive
    games: List[Game]
    max_score: Unsigned
    mean_score: Unsigned
    median_score: Unsigned
    max_tile: Positive
    median_tile: Positive
    mode_tile: Positive

    class Config:
        """Pydantic config."""

        allow_mutation = False


@app.get('/play/batch', response_model=Batch)
def play_batch(spec: BatchSpec):
    """Play batch of 2048."""
    return spec.play()
