import math
import torch
import time
from .game_logic import Game


class Agent(object):

    def __init__(self, device, model):
        self._device = device
        self._model = model.to(device)

    def _eps_func(self, index, eps_start, eps_end, eps_decay):
        return eps_end + (eps_start - eps_end) * math.exp(-1.0 * index / eps_decay)

    def _train_on_turn(self, index, game, num_games, batch_size, eps_start, eps_end, eps_decay, discount, loss_func, optimizer):
        # Calculate q value for each action from state
        state_values = self._model(game.state.flatten(1).float())
        # Flag futile actions
        state_values[game.available_actions().logical_not()] = -1
        # Choose action with max q_value
        actions = state_values.max(1)[1]
        # Choose random available action
        rand_actions = (game.available_actions() * torch.rand((batch_size, 4), device=self._device)).max(1)[1]
        # Mix policy and exploration
        eps = self._eps_func(index, eps_start, eps_end, eps_decay)
        rand_indices = (torch.rand(batch_size, device=self._device) < eps)
        actions[rand_indices] = rand_actions[rand_indices]
        # Calculate q-values
        q_values = state_values.gather(1, actions.unsqueeze(1)).squeeze()
        # Do action
        rewards = game.do_action(actions.to(torch.int8), 1)
        rewards[game.game_over(dim=1)] = 0
        # Calculate Q-value of the next state
        with torch.no_grad():
            next_state_values = self._model(game.state.flatten(1).float())
        next_state_values[game.available_actions().logical_not()] = -1
        next_q_values = next_state_values.max(1)[0]
        next_q_values[game.game_over(dim=1)] = 0
        # Calculate expected Q-values
        expected_q_values = (next_q_values * discount) + rewards
        # Calculate loss
        loss = loss_func(q_values, expected_q_values)
        # Backpropogate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _train_on_game(self, index, game, num_games, batch_size, eps_start, eps_end, eps_decay, discount, loss_func, optimizer):
        while not game.game_over():
            self._train_on_turn(index, game, num_games, batch_size, eps_start, eps_end, eps_decay, discount, loss_func, optimizer)

    def init_training_session(self, num_games, batch_size, eps_start, eps_end, eps_decay, discount, loss_func, optimizer):
        start_time = time.time()
        for index in range(num_games):
            game = Game(batch_size, 4)
            self._train_on_game(index, game, num_games, batch_size, eps_start, eps_end, eps_decay, discount, loss_func, optimizer)
            game_time = time.time() - start_time
            mean = float(game.score.float().mean())
            std = float(game.score.float().std())
            yield index, game_time, mean, std
