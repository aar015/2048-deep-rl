"""deep2048 numeric types."""
from pydantic import conint, confloat

Fraction = confloat(ge=0, le=1)
Positive = conint(gt=0)
Unsigned = conint(ge=0)
