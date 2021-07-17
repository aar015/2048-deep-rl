"""Import all api routes to run with uvicorn."""
from .app import app  # noqa
from . import game, random  # noqa
