"""deep2048 engine router."""
from fastapi import APIRouter
from functools import partial
from jax import numpy as jnp
from ..agent.agent import Agent
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
    if spec.agent is not None:
        # update q-values using agent
        pass
    elif spec.games[0].q is not None:
        q_values = jnp.ndarray([game.q for game in spec.games])
    else:
        q_values = None
    agent = Agent()
    engine.next(
        choose_actions=partial(
            agent.choose,
            user_choices=[game.action for game in spec.games],
            q_values=q_values
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
    if spec.agent is not None:
        # update q-values using agent
        pass
    else:
        choose_actions = Agent().choose
    engine.run(choose_actions=choose_actions)
    return EngineState.from_engine(engine, spec.agent)


@router.put('/run', response_model=EngineState)
def run_engine_from_init_state(spec: EngineInit):
    """Intialize engine and advance to last state."""
    engine = Engine(key=spec.key.array, n=spec.n)
    if spec.agent is not None:
        # update q-values using agent
        pass
    else:
        choose_actions = Agent().choose
    engine.run(choose_actions=choose_actions)
    return EngineState.from_engine(engine, spec.agent)
