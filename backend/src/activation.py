"""Define activation functions for network."""
from enum import Enum
from jax import nn
from typing import Callable


class Activation(str, Enum):
    """Activation string enum for typing."""

    relu = 'relu'
    relu6 = 'relu6'
    sigmoid = 'sigmoid'
    softplus = 'softplus'
    soft_sign = 'soft-sign'
    silu = 'silu'
    swish = 'swish'
    log_sigmoid = 'log-sigmoid'
    leaky_relu = 'leaky-relu'
    hard_sigmoid = 'hard-sigmoid'
    hard_silu = 'hard-silu'
    hard_swish = 'hard-swish'
    hard_tanh = 'hard-tanh'
    elu = 'elu'
    celu = 'celu'
    selu = 'selu'
    gelu = 'gelu'
    glu = 'glu'


def activation_func(activation: Activation) -> Callable:
    """Convert activation string to activation function."""
    return getattr(nn, activation.replace('-', '_'))
