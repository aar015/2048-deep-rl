"""Something."""


class Agent(object):
    """Something."""

    def __init__(self, value_func):
        """Something."""
        self._value_func = value_func

    @property
    def value_func(self):
        """Something."""
        return self._value_func

    def choose_k(self, game):
        """Something."""
        rotations = game.rotations()
        valid_rotations = game._can_advance(rotations)
        values = self.value_func.evaluate(rotations) * valid_rotations
        k = values.argmax(dim=1)
        return k

    def play(self, game):
        """Something."""
        while not game.game_over():
            k = self.choose_k(game)
            game.rotate(k)
            game.advance()
            game.add_tile()
        game.history.append(game)
        return game.score

    def train(self, sample):
        state = sample.state
        target = sample.final_score - sample.score
        self.value_func.update(state, target)
