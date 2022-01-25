"""deep2048 engine router."""
from fastapi import APIRouter
from functools import partial
from jax import numpy as jnp
from jax.random import uniform
from ..engine.batch import Batch
from ..engine.engine import Engine
from ..specs.engine import EngineInit, EngineState, EngineAction

router = APIRouter()


@router.put('/init', response_model=EngineState)
def init_engine_state(spec: EngineInit):
    """Initialize engine state."""
    engine = Engine(key=spec.key.array, n=spec.n)
    return EngineState.from_engine(engine, spec.agent)


@router.put('/next', response_model=EngineState)
def next_engine_state(spec: EngineAction):
    """Advance engine to next state."""
    engine = Engine(
        key=spec.key.array,
        batch=Batch(string=[game.state for game in spec.games]),
        scores=jnp.array([game.score for game in spec.games], jnp.uint32)
    )
    engine.next(
        choose_actions=partial(
            choose_actions,
            user_actions=[game.action for game in spec.games],
            agent=spec.agent
        )
    )
    return EngineState.from_engine(engine, spec.agent)


@router.put('/last', response_model=EngineState)
def last_engine_state(spec: EngineState):
    """Advance engine to last state."""
    engine = Engine(
        key=spec.key.array,
        batch=Batch(string=[game.state for game in spec.games]),
        scores=jnp.array([game.score for game in spec.games], jnp.uint32)
    )
    engine.run(
        choose_actions=partial(
            choose_actions,
            agent=spec.agent
        )
    )
    return EngineState.from_engine(engine, spec.agent)


@router.put('/run', response_model=EngineState)
def run_engine_from_init_state(spec: EngineInit):
    """Intialize engine and advance to last state."""
    engine = Engine(key=spec.key.array, n=spec.n)
    engine.run(
        choose_actions=partial(
            choose_actions,
            agent=spec.agent
        )
    )
    return EngineState.from_engine(engine, spec.agent)


def choose_actions(key, batch, user_actions=None, q_values=None, agent=None):
    """Choose input actions based on user input, q-values, agent, or random."""
    # Implement q-values
    actions = jnp.argmax(
        batch.valid_actions * uniform(key, (batch.n, 4), minval=0.01), axis=1
    ) if agent is None else agent.choose(key, batch)
    return jnp.array([
        user_action if user_action is not None else action
        for user_action, action in zip(user_actions, actions)
    ]) if user_actions is not None else actions
