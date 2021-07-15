"""Define board in 2048 game."""
import torch
from .app import app
from .primatives import uint4
from functools import partial
from pydantic import BaseModel
from pydantic.types import constr


class Board(BaseModel):
    """Model of 2048 game board."""

    state: constr(strip_whitespace=True, to_lower=True, min_length=16,
                  max_length=16, regex='^[a-f0-9]*$')


class LongBoard(BaseModel):
    """Long form model of 2048 game board."""

    tile0: uint4
    tile1: uint4
    tile2: uint4
    tile3: uint4
    tile4: uint4
    tile5: uint4
    tile6: uint4
    tile7: uint4
    tile8: uint4
    tile9: uint4
    tilea: uint4
    tileb: uint4
    tilec: uint4
    tiled: uint4
    tilee: uint4
    tilef: uint4


def tensor_to_int(state: torch.Tensor):
    """Convert 4x4 pytorch tensor to int."""
    result = 0
    for x, row in enumerate(state):
        for y, tile in enumerate(row):
            result += int(tile) << (60 - (x << 4) - (y << 2))
    return result


def tensor_to_str(state: torch.Tensor):
    """Convert 4x4 pytorch tensor to board state string."""
    num = tensor_to_int(state)
    return f'{num:0>16x}'


def str_to_list(state: str):
    """Convert board state string to python list."""
    state = Board(state=state)
    return list(map(partial(int, base=16), list(state.state)))


def str_to_long_model(state: str):
    """Convert board state string to long pydantic model."""
    state = str_to_list(state)
    state = {f'tile{i:x}': tile for i, tile in enumerate(state)}
    state = LongBoard(**state)
    return state


def str_to_tensor(state: str):
    """Convert board state string to pytorch tensor."""
    state = str_to_list(state)
    state = torch.tensor(state).reshape((4, 4))
    return state


@app.get('/board', response_model=Board)
def route_long_board_to_board(
    tile0: uint4, tile1: uint4, tile2: uint4, tile3: uint4,
    tile4: uint4, tile5: uint4, tile6: uint4, tile7: uint4,
    tile8: uint4, tile9: uint4, tilea: uint4, tileb: uint4,
    tilec: uint4, tiled: uint4, tilee: uint4, tilef: uint4
):
    """Convert list of tiles to board string."""
    tensor = torch.tensor([
        tile0, tile1, tile2, tile3,
        tile4, tile5, tile6, tile7,
        tile8, tile9, tilea, tileb,
        tilec, tiled, tilee, tilef
    ]).reshape(4, 4)
    string = tensor_to_str(tensor)
    board = Board(state=string)
    return board


@app.get('/board/{state}', response_model=LongBoard)
def route_board_to_long_board(state: str):
    """Convert board string to list of tiles."""
    return str_to_long_model(state)
