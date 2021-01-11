import random
import torch
from collections import namedtuple
from itertools import chain

Sample = namedtuple('Sample',
                    ['turn', 'final_turn', 'score',
                     'final_score', 'state', 'rotation'])


class History(object):

    def __init__(self, batch_size, device=torch.device("cpu")):
        self._batch_size = batch_size
        self._device = device
        self._turn = 0
        self._final_turn = torch.zeros(batch_size, device=device,
                                       dtype=torch.int32)
        self._score = torch.tensor((), device=device, dtype=torch.int32)
        self._final_score = torch.zeros(batch_size, device=device,
                                        dtype=torch.int32)
        self._state = torch.tensor((), device=device, dtype=torch.uint8)
        self._rotation = torch.tensor((), device=device, dtype=torch.int32)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def final_score(self):
        return self._final_score

    @property
    def num_turns(self):
        return self._num_turns

    def append(self, game):
        new_score = game.score.unsqueeze(0)
        self._score = torch.cat([self._score, new_score])
        new_state = game.state.unsqueeze(0)
        self._state = torch.cat([self._state, new_state])
        new_rotation = game.rotation.unsqueeze(0)
        self._rotation = torch.cat([self._rotation, new_rotation])
        state_change = self._final_turn.logical_not()
        game_over = game.game_over(dim=1)
        new_game_over = torch.logical_and(state_change, game_over)
        self._final_turn[new_game_over] = self._turn
        self._final_score[new_game_over] = game.score[new_game_over].clone()
        self._turn += 1

    def sample(self, size):
        indices = [[[i, j] for j in range(self._final_turn[i])]
                   for i in range(self.batch_size)]
        indices = list(chain.from_iterable(indices))
        indices = random.sample(indices, 100)
        indices = torch.tensor(indices)
        return Sample(
            turn=indices[:, 1],
            final_turn=self._final_turn[indices[:, 0]],
            score=self._score[indices[:, 1], indices[:, 0]],
            final_score=self._final_score[indices[:, 0]],
            state=self._state[indices[:, 1], indices[:, 0]],
            rotation=self._rotation[indices[:, 1], indices[:, 0]]
        )
