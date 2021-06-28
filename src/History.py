"""Fill."""
import random
import torch
from collections import namedtuple
from itertools import chain

Record = namedtuple(
    'Record',
    ['turn', 'final_turn', 'score', 'final_score', 'state', 'rotation']
)


class Log(object):
    """Something."""

    def __init__(self, size, device):
        """Something."""
        self._size = size
        self._device = device
        self._turn = 0
        self._num_turns = torch.zeros(size, device=device, dtype=torch.int32)
        self._states = torch.tensor((), device=device, dtype=torch.uint8)
        self._scores = torch.tensor((), device=device, dtype=torch.int32)
        self._final_scores = torch.zeros(
            size, device=device, dtype=torch.int32)
        self._rotations = torch.tensor((), device=device, dtype=torch.int32)

    @property
    def size(self):
        """Something."""
        return self._size

    @property
    def device(self):
        """Something."""
        return self._device

    @property
    def num_turns(self):
        """Something."""
        return self._num_turns

    @property
    def final_scores(self):
        """Something."""
        return self._final_scores

    def append(self, batch):
        """Something."""
        new_scores = batch.score.unsqueeze(0)
        self._scores = torch.cat([self._scores, new_scores])
        new_states = batch.state.unsqueeze(0)
        self._states = torch.cat([self._states, new_states])
        new_rotations = batch.rotation.unsqueeze(0)
        self._rotations = torch.cat([self._rotations, new_rotations])
        state_change = self._final_turn.logical_not()
        game_over = batch.game_over(dim=1)
        new_game_over = torch.logical_and(state_change, game_over)
        self._final_turn[new_game_over] = self._turn
        self._final_score[new_game_over] = batch.score[new_game_over].clone()
        self._turn += 1

    def sample(self, size):
        """Something."""
        indices = [[[i, j] for j in range(self._final_turn[i])]
                   for i in range(self.batch_size)]
        indices = list(chain.from_iterable(indices))
        indices = random.sample(indices, 100)
        indices = torch.tensor(indices)
        return Record(
            turn=indices[:, 1],
            final_turn=self._final_turn[indices[:, 0]],
            score=self._score[indices[:, 1], indices[:, 0]],
            final_score=self._final_score[indices[:, 0]],
            state=self._state[indices[:, 1], indices[:, 0]],
            rotation=self._rotation[indices[:, 1], indices[:, 0]]
        )
