"""Define board in 2048 game."""
import torch
from ..app import app
from functools import partial
from pydantic import BaseModel
from pydantic.types import conint, constr


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
    return f'{num:x}'


class Board(BaseModel):
    """Model of 2048 game board."""

    state: constr(strip_whitespace=True, min_length=16, max_length=16)


def str_to_list(state: str):
    """Convert board state string to python list."""
    state = Board(state=state)
    return list(map(partial(int, base=16), list(state.state)))


class BoardLong(BaseModel):
    """Long form model of 2048 game board."""

    tile0: conint(ge=0, le=15)
    tile1: conint(ge=0, le=15)
    tile2: conint(ge=0, le=15)
    tile3: conint(ge=0, le=15)
    tile4: conint(ge=0, le=15)
    tile5: conint(ge=0, le=15)
    tile6: conint(ge=0, le=15)
    tile7: conint(ge=0, le=15)
    tile8: conint(ge=0, le=15)
    tile9: conint(ge=0, le=15)
    tilea: conint(ge=0, le=15)
    tileb: conint(ge=0, le=15)
    tilec: conint(ge=0, le=15)
    tiled: conint(ge=0, le=15)
    tilee: conint(ge=0, le=15)
    tilef: conint(ge=0, le=15)


def str_to_long_model(state: str):
    """Convert board state string to long pydantic model."""
    state = str_to_list(state)
    state = {f'tile{i:x}': tile for i, tile in enumerate(state)}
    state = BoardLong(**state)
    return state


@app.get('/environ/board/{state}', response_model=BoardLong)
def board(state: str):
    """Get board tiles from board string."""
    tensor = str_to_tensor(state)
    print(tensor)
    back_int = tensor_to_int(tensor)
    print(back_int)
    back_str = tensor_to_str(tensor)
    print(back_str)
    return str_to_long_model(state)


def str_to_tensor(state: str):
    """Convert board state string to pytorch tensor."""
    state = str_to_list(state)
    state = torch.tensor(state).reshape((4, 4))
    return state
