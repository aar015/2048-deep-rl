"""Something."""
import torch
from numba import cuda, guvectorize


ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


@cuda.jit(device=True)
def _count_zeros(state):
    num_zero = 0
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y] == 0:
                num_zero += 1
    return num_zero


@cuda.jit(device=True)
def _choose_zero(state, nth_zero):
    count = 0
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y] == 0:
                if count == nth_zero:
                    return x, y
                count += 1
    return 0, 0


@cuda.jit(device=True)
def _add_tile(state, random1, random2):
    num_zero = _count_zeros(state)
    if num_zero != 0:
        nth_zero = int(num_zero * random1)
        value = 1 if random2 < 0.9 else 2
        x, y = _choose_zero(state, nth_zero)
        state[x, y] = value
    return num_zero != 0


@cuda.jit(device=True)
def _rotate_index(x, y, rotation, board_size):
    num_rotate = rotation % 4
    while num_rotate > 0:
        x, y = y, board_size - x - 1
        num_rotate -= 1
    return x, y


@cuda.jit(device=True)
def _do_action(state, action, punishment, simulate):
    reward = 0
    update = False
    for x in range(state.shape[0]):
        for y1 in range(state.shape[1] - 1):
            for y2 in range(y1 + 1, state.shape[1]):
                rot_x1, rot_y1 = _rotate_index(x, y1, action, state.shape[0])
                rot_x2, rot_y2 = _rotate_index(x, y2, action, state.shape[0])
                if state[rot_x2, rot_y2] == 0:
                    continue
                elif state[rot_x1, rot_y1] == 0:
                    if not simulate:
                        state[rot_x1, rot_y1] = state[rot_x2, rot_y2]
                        state[rot_x2, rot_y2] = 0
                    update |= True
                else:
                    if state[rot_x1, rot_y1] == state[rot_x2, rot_y2]:
                        if not simulate:
                            state[rot_x1, rot_y1] += 1
                            state[rot_x2, rot_y2] = 0
                        reward += 2 ** state[rot_x1, rot_y1]
                        update |= True
                    break
    return reward if update else -1 * punishment


@guvectorize(['void(u1[:,:], f4[:], f4[:], b1[:])'], '(n,n),(),()->()', target='cuda', nopython=True)
def _add_tiles(state, random1, random2, res):
    res[0] = _add_tile(state, random1[0], random2[0])


@guvectorize(['void(u1[:,:], i1[:], i4, b1, i4[:])'], '(n,n),(),(),()->()', target='cuda', nopython=True)
def _do_actions(state, action, punishment, simulate, reward):
    reward[0] = _do_action(state, action[0], punishment, simulate)


class Game(object):
    """Something."""

    def __init__(self, batch_size, board_size):
        """Something."""
        self._batch_size = batch_size
        self._board_size = board_size
        if batch_size == 0:
            self._state = torch.cuda.ByteTensor(torch.Size((board_size, board_size)))
            self._score = 0
        else:
            self._state = torch.cuda.ByteTensor(torch.Size((batch_size, board_size, board_size)))
            self._score = torch.cuda.IntTensor(batch_size)
            torch.zeros(self._score.shape, dtype=torch.int32, out=self._score)
        torch.zeros(self._state.shape, dtype=torch.uint8, out=self._state)
        self._add_tile()
        self._add_tile()

    @property
    def state(self):
        """Something."""
        return self._state

    @property
    def score(self):
        """Something."""
        return self._score

    @property
    def board_size(self):
        """Something."""
        return self._board_size

    @property
    def batch_size(self):
        """Something."""
        return self._batch_size

    def game_over(self, **kwargs):
        """Something."""
        return self.available_actions().any(**kwargs).logical_not()

    def available_actions(self):
        """Something."""
        if self.batch_size == 0:
            available = torch.cuda.BoolTensor(4)
            for action in range(4):
                reward = _do_actions(self._state, action, 1, True)[0]
                available[action] = True if (reward >= 0) else False
        else:
            available = torch.cuda.BoolTensor(torch.Size((self.batch_size, 4)))
            reward = torch.cuda.IntTensor(self.batch_size)
            for action in range(4):
                _do_actions(self.state, action, 1, True, out=reward)
                available[:, action] = reward >= 0
        return available

    def do_action(self, action, punishment):
        """Something."""
        if self.batch_size == 0:
            reward = _do_actions(self._state, action, punishment, False)[0]
            self._score += max(reward, 0)
        else:
            reward = torch.cuda.IntTensor(self.batch_size)
            _do_actions(self._state, action, punishment, False, out=reward)
            self._score[reward > 0] += reward[reward > 0]
        self._add_tile()
        return reward

    def _add_tile(self):
        if self.batch_size == 0:
            random1 = torch.rand((), dtype=torch.float32)
            random2 = torch.rand((), dtype=torch.float32)
        else:
            random1 = torch.cuda.FloatTensor(self.batch_size)
            random2 = torch.cuda.FloatTensor(self.batch_size)
            torch.rand(random1.shape, dtype=torch.float32, out=random1)
            torch.rand(random2.shape, dtype=torch.float32, out=random2)
        _add_tiles(self._state, random1, random2)