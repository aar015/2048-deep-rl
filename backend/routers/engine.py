"""deep2048 engine router."""
from ..engine.engine import init
from ..specs.engine import EngineInit, EngineState, EngineAction
from fastapi import APIRouter

router = APIRouter()


@router.put('/init', response_model=EngineState)
def init_engine_state(spec: EngineInit):
    """Initialize engine state."""
    return next(init(spec))


@router.put('/next', response_model=EngineState)
def next_engine_state(spec: EngineAction):
    """Advance engine to next state."""
    engine = init(spec.state)
    next(engine)
    actions = [game.action for game in spec.games]
    return engine.send(actions)


@router.put('/last', response_model=EngineState)
def last_engine_state(spec: EngineState):
    """Advance engine to last state."""
    for state in init(spec):
        pass
    return state


@router.put('/run', response_model=EngineState)
def run_engine_from_init_state(spec: EngineInit):
    """Intialize engine and advance to last state."""
    for state in init(spec):
        pass
    return state
