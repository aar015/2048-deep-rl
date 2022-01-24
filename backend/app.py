"""deep2048 web app backend."""
from .routers import engine, random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['http://localhost:3000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(random.router, prefix='/random')
app.include_router(engine.router, prefix='/engine')
