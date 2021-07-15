"""Define pydantic primative types."""
from pydantic import conint


uint4 = conint(ge=0, le=15)
