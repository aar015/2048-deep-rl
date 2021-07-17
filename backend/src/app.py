"""Define FastAPI object."""
import jax
from fastapi import FastAPI
from pydantic import BaseSettings

app = FastAPI()


class Settings(BaseSettings):
    """Pydantic model of app settings."""

    device: str = jax.default_backend()
    target: str = 'cuda' if jax.default_backend() == 'gpu' else 'cpu'

    class Config:
        """Pydantic config."""

        allow_mutation = False


SETTINGS = Settings()


@app.get('/settings/gpu')
def check_gpu_enabled() -> bool:
    """Check if gpu is enabled."""
    return SETTINGS.device == 'gpu'
