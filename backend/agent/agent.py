"""deep2048 agent."""
from jax import numpy as jnp
from jax.random import split, uniform


class Agent(object):
    """Abstract Agent to Play 2048."""

    def __init__(self):
        """Initialize agent."""
        self._type = 'default'

    @property
    def type(self):
        """Get name."""
        return self._type

    def choose(
        self, key, batch, user_choices=None, q_values=None, exploration=None
    ):
        """Choose actions to perform on batch."""
        subkey1, subkey2 = split(key)
        explore = exploration is not None and \
            uniform(subkey1, ()) < exploration
        if q_values is None or explore:
            actions = jnp.argmax(
                batch.valid_actions *
                uniform(subkey2, (batch.n, 4), minval=1e-3),
                axis=1,
            )
        else:
            actions = jnp.argmax(
                (q_values + 1e-3) * batch.valid_actions *
                uniform(subkey2, (batch.n, 4), minval=1e-3),
                axis=1,
            )
        if user_choices is None:
            return actions
        elif type(user_choices) == jnp.ndarray:
            return user_choices
        else:
            return jnp.array([
                user_choice if user_choice is not None else action
                for user_choice, action in zip(user_choices, actions)
            ])
