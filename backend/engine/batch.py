"""deep2048 batch of game states."""
from jax import random
from jax import numpy as jnp
from .actions import init, rotations, choose_rotation, validate, next, add_tile
from .transforms import string_to_small, small_to_string, small_to_medium,\
    small_to_large, medium_to_small, large_to_small


class Batch(object):
    """Batch of 2048 board states."""

    def __init__(
        self, key=None, n=1, string=None, small=None, medium=None, large=None
    ):
        """Initialize states."""
        self._n = None
        self._string = None
        self._small = None
        self._medium = None
        self._large = None
        self._valid_mask = None
        self._valid = None
        self._rotations = None
        self._valid_actions = None
        self._terminal = None
        self._live = None
        self._next = None
        self._reward = None

        if key is not None:
            self._n = n
            keys = random.split(key, n)
            self._large = init(keys)
        elif string is not None:
            self._string = string
        elif small is not None:
            self._small = small
        elif medium is not None:
            self._medium = medium
        elif large is not None:
            self._large = large

        if (
            self._string is None and self._small is None
            and self._medium is None and self._large is None
        ):
            raise Exception('Failed to initialize states.')

    @property
    def n(self) -> int:
        """Get number of states."""
        if self._n is not None:
            return self._n
        if self._small is not None:
            self._n = self._small.shape[0]
            return self.n
        if self._medium is not None:
            self._n = self._medium.shape[0]
            return self.n
        if self._large is not None:
            self._n = self._large.shape[0]
            return self.n
        if self._string is not None:
            self._n = len(self._string)
            return self.n
        raise Exception('No state data.')

    @property
    def string(self):
        """Get states as list of strings."""
        if self._string is not None:
            return self._string
        if self._small is not None:
            self._string = small_to_string(self._small)
            return self.string
        if self._medium is not None:
            self._small = medium_to_small(self._medium)
            return self.string
        if self._large is not None:
            self._small = large_to_small(self._large)
            return self.string
        raise Exception('No state data.')

    @property
    def small(self):
        """Get states as array of shape (n, 2) of type uint32."""
        if self._small is not None:
            return self._small
        if self._medium is not None:
            self._small = medium_to_small(self._medium)
            return self.small
        if self._large is not None:
            self._small = large_to_small(self._large)
            return self.small
        if self._string is not None:
            self._small = string_to_small(self._string)
            return self.small
        raise Exception('No state data.')

    @property
    def medium(self):
        """Get states as array of shape (n, 2, 8) of type uint8."""
        if self._medium is not None:
            return self._medium
        if self._small is not None:
            self._medium = small_to_medium(self._small)
            return self.medium
        if self._large is not None:
            self._small = large_to_small(self._large)
            return self.medium
        if self._string is not None:
            self._small = string_to_small(self._string)
            return self.medium
        raise Exception('No state data.')

    @property
    def large(self):
        """Get states as array of shape (n, 4, 4) of type uint8."""
        if self._large is not None:
            return self._large
        if self._small is not None:
            self._large = small_to_large(self._small)
            return self.large
        if self._medium is not None:
            self._small = medium_to_small(self._medium)
            return self.large
        if self._string is not None:
            self._small = string_to_small(self._string)
            return self.large
        raise Exception('No state data.')

    @property
    def valid_mask(self):
        """Get mask of states that pass validation."""
        if self._valid_mask is not None:
            return self._valid_mask
        self._valid_mask = validate(self.large)
        return self.valid_mask

    @property
    def valid(self):
        """Get valid states."""
        if self._valid is not None:
            return self._valid
        self._valid = Batch(large=self.large[self.valid_mask])
        return self.valid

    @property
    def rotations(self):
        """Get rotations of states."""
        if self._rotations is not None:
            return self._rotations
        self._rotations = rotations(self.large)
        return self.rotations

    @property
    def valid_actions(self):
        """Get valid actions for state."""
        if self._valid_actions is not None:
            return self._valid_actions
        self._valid_actions = validate(self.rotations)
        return self.valid_actions

    @property
    def terminal(self):
        """Get mask of terminal states."""
        if self._terminal is not None:
            return self._terminal
        self._terminal = jnp.logical_not(self.valid_actions.any(axis=1))
        return self.terminal

    @property
    def live(self):
        """Get live states."""
        if self._live is not None:
            return self._live
        self._live = Batch(large=self.large[jnp.logical_not(self.terminal)])
        return self.live

    @property
    def next(self):
        """Get next states."""
        if self._next is not None:
            return self._next
        nextStates = self.large + 0
        self._reward = next(nextStates)
        self._next = Batch(large=nextStates)
        return self.next

    @property
    def reward(self):
        """Get rewards for advancing states."""
        if self._reward is not None:
            return self._reward
        self.next
        return self.reward

    def rotate(self, actions):
        """Rotate states."""
        return Batch(large=choose_rotation(self.rotations, actions))

    def add_tile(self, key):
        """Add new tile to board."""
        key1, key2 = random.split(key)
        rand1 = random.uniform(key1, (self.n,))
        rand2 = random.uniform(key2, (self.n,))
        nextStates = self.large + 0
        success = add_tile(nextStates, rand1, rand2)
        if not jnp.any(success):
            raise Exception('Adding tile did not change any state.')
        return Batch(large=nextStates)
