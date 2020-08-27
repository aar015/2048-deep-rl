"""Code Implementing Game Logic."""
import torch
from numba import guvectorize


LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3


def _check_action(state, action, available):
    available[0] = False
    for x in range(state.shape[0]):
        for y1 in range(state.shape[1] - 1):
            for y2 in range(y1 + 1, state.shape[1]):
                # Rotatate Coordinates to Reflect Action to Be Taken
                rot_x1, rot_y1 = x, y1
                rot_x2, rot_y2 = x, y2
                num_rotate = action[0] % 4
                while num_rotate > 0:
                    rot_x1, rot_y1 = rot_y1, state.shape[0] - rot_x1 - 1
                    rot_x2, rot_y2 = rot_y2, state.shape[0] - rot_x2 - 1
                    num_rotate -= 1
                # Check if Action is Available
                if state[rot_x2, rot_y2] == 0:
                    continue
                elif state[rot_x1, rot_y1] == 0:
                    available[0] = True
                else:
                    if state[rot_x1, rot_y1] == state[rot_x2, rot_y2]:
                        available[0] = True
                    break


def _do_action(state, action, reward):
    reward[0] = 0
    for x in range(state.shape[0]):
        for y1 in range(state.shape[1] - 1):
            for y2 in range(y1 + 1, state.shape[1]):
                # Rotatate Coordinates to Reflect Action to Be Taken
                rot_x1, rot_y1, rot_x2, rot_y2 = x, y1, x, y2
                num_rotate = action[0] % 4
                while num_rotate > 0:
                    rot_x1, rot_y1 = rot_y1, state.shape[0] - rot_x1 - 1
                    rot_x2, rot_y2 = rot_y2, state.shape[0] - rot_x2 - 1
                    num_rotate -= 1
                # Perform Action
                if state[rot_x2, rot_y2] == 0:
                    continue
                elif state[rot_x1, rot_y1] == 0:
                    state[rot_x1, rot_y1] = state[rot_x2, rot_y2]
                    state[rot_x2, rot_y2] = 0
                else:
                    if state[rot_x1, rot_y1] == state[rot_x2, rot_y2]:
                        state[rot_x1, rot_y1] += 1
                        state[rot_x2, rot_y2] = 0
                        reward[0] += 2 ** state[rot_x1, rot_y1]
                    break


def _add_tile(state, random1, random2, empty_spaces):
    # Count Number of Empty Tiles
    num_zero = 0
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y] == 0:
                num_zero += 1
    if num_zero != 0:
        # Find Nth Empty Tile
        nth_zero = int(num_zero * random1[0])
        count = 0
        for _x in range(state.shape[0]):
            for _y in range(state.shape[1]):
                if state[_x, _y] == 0:
                    if count == nth_zero:
                        x, y = _x, _y
                    count += 1
        # Update Empty Tile
        state[x, y] = 1 if random2[0] < 0.9 else 2
    empty_spaces[0] = num_zero - 1


class Game(object):
    """Something."""

    def __init__(self, batch_size=0, board_size=4, device=torch.device("cpu"), target=None):
        """Something."""
        # Save Parameters
        self._batch_size = batch_size
        self._board_size = board_size
        self._device = device
        if target is None:
            self._target = self.device.type
        else:
            self._target = target
        # Build Accelerated Functions
        external_func = {'check_action': _check_action, 'do_action': _do_action, 'add_tile': _add_tile}
        func_decorator = {
            'check_action': guvectorize(['void(u1[:,:], i1[:], u1[:])'], '(n,n),()->()',
                                        target=self.target, nopython=True),
            'do_action': guvectorize(['void(u1[:,:], i1[:], i4[:])'], '(n,n),()->()',
                                     target=self.target, nopython=True),
            'add_tile': guvectorize(['void(u1[:,:], f4[:], f4[:], i1[:])'], '(n,n),(),()->()',
                                    target=self.target, nopython=True),
        }
        self._accelerated_func = {}
        for key in external_func.keys():
            self._accelerated_func[key] = func_decorator[key](external_func[key])
        # Initialize Data
        if batch_size == 0:
            self._state = torch.zeros((self.board_size, self.board_size), dtype=torch.uint8, device=self.device)
            self._score = 0
        else:
            self._state = torch.zeros((self.batch_size, self.board_size, self.board_size),
                                      dtype=torch.uint8, device=self.device)
            self._score = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.add_tile()
        self.add_tile()

    @property
    def batch_size(self):
        """Something."""
        return self._batch_size

    @property
    def board_size(self):
        """Something."""
        return self._board_size

    @property
    def device(self):
        """Something."""
        return self._device

    @property
    def target(self):
        """Something."""
        return self._target

    @property
    def _check_action(self):
        """Something."""
        return self._accelerated_func['check_action']

    @property
    def _do_action(self):
        """Something."""
        return self._accelerated_func['do_action']

    @property
    def _add_tile(self):
        """Something."""
        return self._accelerated_func['add_tile']

    @property
    def state(self):
        """Something."""
        return self._state

    @property
    def score(self):
        """Something."""
        return self._score

    def new_game(self):
        """Something."""
        self._state *= 0
        self._score *= 0
        self.add_tile()
        self.add_tile()

    def game_over(self, **kwargs):
        """Something."""
        return self.available_actions().any(**kwargs).logical_not()

    def available_actions(self):
        """Something."""
        if self.batch_size == 0:
            available = torch.zeros(4, dtype=torch.bool, device=self.device)
            for action in range(4):
                if self.device.type == 'cpu':
                    available[action] = self._check_action(self._state, action).item()
                else:
                    available[action] = self._check_action(self._state, action)[0]
        else:
            available = torch.zeros((self.batch_size, 4), dtype=torch.uint8, device=self.device)
            for action in range(4):
                if self.device.type == 'cpu':
                    self._check_action(self._state, action, out=available[:, action].numpy())
                else:
                    self._check_action(self._state, action, out=available[:, action])
        return available

    def check_action(self, action):
        """Something."""
        if self.batch_size == 0:
            if self.device.type == 'cpu':
                available = self._check_action(self._state, action).item()
            else:
                available = self._check_action(self._state, action)[0]
        else:
            available = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)
            if self.device.type == 'cpu':
                self._check_action(self._state, action, out=available.numpy())
            else:
                self._check_action(self._state, action, out=available)
        return available

    def do_action(self, action):
        """Something."""
        if self.batch_size == 0:
            if self.device.type == 'cpu':
                reward = self._do_action(self._state, action).item()
            else:
                reward = self._do_action(self._state, action)[0]
        else:
            reward = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
            if self.device.type == 'cpu':
                self._do_action(self._state, action, out=reward.numpy())
            else:
                self._do_action(self._state, action, out=reward)
        self._score += reward
        return reward

    def simulate_action(self, action):
        """Something."""
        state_copy = self.state.clone()
        if self.batch_size == 0:
            if self.device.type == 'cpu':
                reward = self._do_action(state_copy, action).item()
            else:
                reward = self._do_action(state_copy, action)[0]
        else:
            reward = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
            if self.device.type == 'cpu':
                self._do_action(state_copy, action, out=reward.numpy())
            else:
                self._do_action(state_copy, action, out=reward)
        return reward, state_copy

    def add_tile(self):
        """Something."""
        if self.batch_size == 0:
            random1 = torch.rand((), dtype=torch.float32, device=self.device)
            random2 = torch.rand((), dtype=torch.float32, device=self.device)
            if self.device.type == 'cpu':
                success = self._add_tile(self._state, random1, random2).item()
            else:
                success = self._add_tile(self._state, random1, random2)[0]
        else:
            random1 = torch.rand(self.batch_size, dtype=torch.float32, device=self.device)
            random2 = torch.rand(self.batch_size, dtype=torch.float32, device=self.device)
            success = torch.zeros(self.batch_size, dtype=torch.int8, device=self.device)
            if self.device.type == 'cpu':
                self._add_tile(self._state, random1, random2, out=success.numpy())
            else:
                self._add_tile(self._state, random1, random2, out=success)
        return success