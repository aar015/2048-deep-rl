"""Define FastAPI object."""
import jax
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings

app = FastAPI()

origins = ['http://localhost:3000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
