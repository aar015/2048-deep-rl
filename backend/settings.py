"""deep2048 web app settings."""
import jax
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Pydantic model of app settings."""

    device: str = jax.default_backend()
    target: str = 'cuda' if jax.default_backend() == 'gpu' else 'cpu'

    class Config:
        """Pydantic config."""

        allow_mutation = False


SETTINGS = Settings()
