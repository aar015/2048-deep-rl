"""Something."""
import torch
import time
import math


class Agent(object):
    """Something."""

    def __init__(self, value_func):
        """Something."""
        self._value_func = value_func

    @property
    def value_func(self):
        """Something."""
        return self._value_func

    def eps_func(self, index, eps_start, eps_end, eps_decay):
        """Something."""
        return eps_end + (eps_start - eps_end) * math.exp(-1.0 * index / eps_decay)

    def choose_action(self, game, eps=0):
        """Something."""
        if game.batch_size == 0:
            random1 = torch.rand((), dtype=torch.float32, device=game.device)
            random2 = torch.rand(4, dtype=torch.float32, device=game.device)
            if random1 < eps:
                action = (game.available_actions() * random2).argmax()
            else:
                future_reward = torch.zeros(4, dtype=torch.float32, device=game.device)
                for action in range(4):
                    reward, after_state = game.simulate_action(action)
                    future_reward[action] = reward + self.value_func.evaluate(after_state)
                future_reward[game.available_actions().logical_not()] = -1
                action = future_reward.argmax()
        else:
            random1 = torch.rand(game.batch_size, dtype=torch.float32, device=game.device)
            random2 = torch.rand((game.batch_size, 4), dtype=torch.float32, device=game.device)
            future_reward = torch.zeros((game.batch_size, 4), dtype=torch.float32, device=game.device)
            for action in range(4):
                reward, after_state = game.simulate_action(action)
                future_reward[:, action] = reward + self.value_func.evaluate(after_state)
            future_reward[game.available_actions().logical_not()] = -1
            action = future_reward.max(1)[1]
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
        prev_after_state = None
        while not game.game_over():
            action = self.choose_action(game, eps)
            reward = game.do_action(action)
            after_state = game.state.clone()
            game.add_tile()
            if prev_after_state is not None:
                self.value_func.update(prev_after_state, reward, after_state, **kwargs)
            prev_after_state = after_state
        return game.score

    def init_training_session(self, num_games, game, eps_start, eps_end, eps_decay, **kwargs):
        """Something."""
        start_time = time.time()
        for index in range(num_games):
            game.new_game()
            eps = self.eps_func(index, eps_start, eps_end, eps_decay)
            self.train_on_game(game, eps, **kwargs)
            game_time = time.time() - start_time
            if game.batch_size == 0:
                mean = game.score
                std = 0
            else:
                mean = float(game.score.float().mean())
                std = float(game.score.float().std())
            yield index, game_time, mean, std