"""Define 2048 game."""
from pydantic import BaseModel
from typing import List, Optional


class Turn(BaseModel):
    """Model for Turn object."""

    id: Optional(int)
    turnNumber: int
    currentScore: int
    futureScore: int
    boardState: int
    terminal: bool
    move: Optional(int)
    turnState: Optional(int)

    class Config:
        """Pydantic config class."""


class Game(BaseModel):
    """Model for Game object."""

    id: Optional(int)
    numTurns: int
    finalScore: int
    highTile: int
    turns: List[Turn]

    class Config:
        """Pydantic config class."""


class Batch(BaseModel):
    """Model for Batch object."""

    id: Optional(int)
    size: int
    games: List[Game]

    class Config:
        """Pydantic config class."""


class TrainingSession(BaseModel):
    """Model for Epoch object."""

    id: Optional(int)
    sessionNumber: int
    playedBatch: Batch
    learnedBatches: List[Batch]

    class Config:
        """Pydantic config class."""


class Agent(BaseModel):
    """Model for Agent object."""

    id: int
    network: str
    networkUpdate: Optional(str)
    trainingSessions: Optional(List(TrainingSession))

    class Config:
        """Pydantic config class."""


class Generation(BaseModel):
    """Model for Generation object."""

    id: int

    class Config:
        """Pydantic config class."""


class Experiment(BaseModel):
    """Model for Experiment object."""

    id: int

    class Config:
        """Pydantic config class."""
