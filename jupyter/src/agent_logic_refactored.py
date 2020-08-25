"""Something."""
import torch
import time
import math


class Agent(object):
    """Something."""

    def __init__(self, model, device):
        """Something."""
        self._device = device
        self._model = model.to(device)

    @property
    def model(self):
        """Something."""
        return self._model

    @property
    def device(self):
        """Something."""
        return self._device

    def choose_action(self, game, eps=0):
        """Something."""
        if game.batch_size == 0:
            random1 = torch.rand((), dtype=torch.float32, device=self.device)
            random2 = torch.rand(4, dtype=torch.float32, device=self.device)
            if random1 < eps:
                action = (game.available_actions() * random2).argmax()
            else:
                expected_reward = torch.zeros(4, dtype=torch.float32, device=self.device)
                for action in range(4):
                    expected_reward[action] = self.expected_reward(game, action)
                expected_reward[game.available_actions().logical_not()] = -1
                action = expected_reward.argmax()
        else:
            random1 = torch.rand(game.batch_size, dtype=torch.float32, device=self.device)
            random2 = torch.rand((game.batch_size, 4), dtype=torch.float32, device=self.device)
            expected_reward = torch.zeros((game.batch_size, 4), dtype=torch.float32, device=self.device)
            for action in range(4):
                expected_reward[:, action] = self.expected_reward(game, action)
            expected_reward[game.available_actions().logical_not()] = -1
            action = expected_reward.max(1)[1]
            rand_action = (game.available_actions() * random2).max(1)[1]
            action[random1 < eps] = rand_action[random1 < eps]
        return action.to(torch.int8)

    def play_game(self, game):
        """Something."""
        while not game.game_over():
            action = self.choose_action(game)
            game.do_action(action)
            game.add_tile()
        return game.score

    def train_on_game(self, game, eps, **kwargs):
        """Something."""
        prev_after_state_value = None
        while not game.game_over():
            action = self.choose_action(game, eps)
            reward = game.do_action(action)
            after_state_value = self.evaluate(game.state)
            game.add_tile()
            if prev_after_state_value is not None:
                self.update(prev_after_state_value, reward, after_state_value, **kwargs)
            prev_after_state_value = after_state_value
        return game.score

    def init_training_session(self, num_games, game, eps_func, **kwargs):
        """Something."""
        start_time = time.time()
        for index in range(num_games):
            game.new_game()
            self.train_on_game(game, eps_func(index), **kwargs)
            game_time = time.time() - start_time
            if game.batch_size == 0:
                mean = game.score
                std = 0
            else:
                mean = float(game.score.float().mean())
                std = float(game.score.float().std())
            yield index, game_time, mean, std

    def expected_reward(self, game, action):
        """Something."""
        reward, after_state = game.simulate_action(action)
        return reward + self.evaluate(after_state)

    def evaluate(self, state):
        """Something."""
        return self.model(state.flatten(1).float())

    def eps_func(self, index, eps_start, eps_end, eps_decay):
        """Something."""
        return eps_end + (eps_start - eps_end) * math.exp(-1.0 * index / eps_decay)

    def update()