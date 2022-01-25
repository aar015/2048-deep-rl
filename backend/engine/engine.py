"""deep2048 game engine."""
from .batch import Batch
from .actions import validate
from ..specs.engine import EngineInit, EngineState
from jax import numpy as jnp
from jax.random import uniform
from typing import Union


def init(spec: Union[EngineInit, EngineState]):
    """Initialize 2048 game engine."""
    key = spec.key
    n = spec.n
    agent = spec.agent
    if type(spec) == EngineInit:
        key, subkey = key.split()
        batch = Batch(key=subkey.array, n=spec.n)
        scores = jnp.zeros(spec.n)
        in_actions = yield EngineState.from_array(key, batch, scores, agent)
    elif type(spec) == EngineState:
        strings = [game.state for game in spec.games]
        batch = Batch(string=strings)
        scores = jnp.array([game.score for game in spec.games])
        in_actions = yield EngineState.from_array(key, batch, scores, agent)
    else:
        raise Exception('spec must be of type EngineInit or EngineState.')
    while not jnp.all(batch.terminal):
        key, subkey1, subkey2 = key.split(3)
        actions = jnp.argmax(
            validate(batch.rotations) * uniform(subkey1.array, (n, 4)), axis=1
        ) if agent is None else agent.choose(subkey1.array, batch)
        if in_actions is not None:
            actions = jnp.array([
                in_action if in_action is not None else action
                for in_action, action in zip(in_actions, actions)
            ])
        rotated = batch.rotate(actions)
        batch = rotated.next.add_tile(subkey2.array).rotate(-1 * actions)
        scores = scores + rotated.reward
        in_actions = yield EngineState.from_array(key, batch, scores, agent)
