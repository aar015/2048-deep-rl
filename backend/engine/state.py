"""deep2048 game state."""
from pydantic.types import constr

State = constr(
    strip_whitespace=True, to_lower=True,
    min_length=16, max_length=16, regex='^[a-f0-9]*$'
)
