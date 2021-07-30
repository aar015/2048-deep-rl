"""Define NN model."""
from .activation import Activation, activation_func
from .state import States, _large_to_medium
from functools import partial
from typing import List
from jax import jit, vmap
from jax import numpy as jnp
from jax.nn import relu
from jax.random import split, normal
from pydantic import BaseModel, conint
from typing import Optional

Positive = conint(gt=0)


class Layer(BaseModel):
    """Pydantic model for network layer."""

    n_in: Positive
    n_out: Positive
    weights: List[List[float]]
    biases: List[float]
    activation: Optional[Activation]

    class Config:
        """Pydantic config."""

        allow_mutation = False


class Network(BaseModel):
    """Pydantic model for neural network."""

    __slots__ = ('_params')
    sym_layers: List[Layer]
    asym_layers: List[Layer]

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def __init__(self, **kwargs):
        """Initialize neural network."""
        super().__init__(**kwargs)
        object.__setattr__(self, '_params', None)

    @property
    def params(self):
        """Get parameters corresponding to network."""
        if self._params is not None:
            return self._params
        temp = NetworkParams(**{
            'sym_layers': [{
                'weights': jnp.array(layer.weights),
                'biases': jnp.array(layer.biases),
            } for layer in self.sym_layers],
            'aysm_layers': [{
                'weights': jnp.array(layer.weights),
                'biases': jnp.array(layer.biases),
            } for layer in self.sym_layers],
            'activations': [
                layer.activation for layer in
                self.sym_layers + self.asym_layers[:-1]
            ],
        })
        object.__setattr__(self, '_params', temp)
        return self.params


class LayerParams(BaseModel):
    """Pydantic model for layer parameters."""

    weights: jnp.ndarray
    biases: jnp.ndarray

    class Config:
        """Pydantic config."""

        allow_mutation = False
        arbitrary_types_allowed = True


class NetworkParams(BaseModel):
    """Pydantic model for network parameters."""

    __slots__ = ('_network')
    sym_layers: List[LayerParams]
    asym_layers: List[LayerParams]
    activations: List[Activation]

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def __init__(self, **kwargs):
        """Initialize network parameters."""
        super().__init__(**kwargs)
        object.__setattr__(self, '_network', None)

    @property
    def network(self):
        """Get network corresponding to parameters."""
        if self._network is not None:
            return self._network
        temp = Network(**{
            'sym_layers': [{
                'n_in': layer.weights.shape[0],
                'n_out': layer.weights.shape[1],
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist(),
                'activation': activation,
            } for layer, activation in zip(
                self.sym_layers,
                self.activations[:len(self.sym_layers)]
            )],
            'asym_layers': [{
                'n_in': layer.weights.shape[0],
                'n_out': layer.weights.shape[1],
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist(),
                'activation': activation,
            } for layer, activation in zip(
                self.asym_layers,
                (self.activations[len(self.sym_layers):] + [None])
            )],
        })
        object.__setattr__(self, '_network', temp)
        return self.network

    def predict(self, states: States):
        """Predict future reward for each move."""
        params = self.dict()
        return _predict(
            params['sym_layers'], params['asym_layers'],
            tuple(params['activations']), states.rotations
        )


class LayerDispatch(BaseModel):
    """Pydantic model for outline to build layer."""

    width: Positive
    activation: Activation

    class Config:
        """Pydantic config."""

        allow_mutation = False


class NetworkDispatch(BaseModel):
    """Pydantic model for outline to build neural network."""

    sym_layers: List[LayerDispatch]
    asym_layers: List[LayerDispatch]

    class Config:
        """Pydantic config."""

        allow_mutation = False

    def init_params(self, key: jnp.ndarray, scale: float = 1):
        """Initialize model params."""
        sym_layers, asym_layers = [], []
        for index, (layers, dispatch) in enumerate((
            (sym_layers, self.sym_layers), (asym_layers, self.asym_layers)
        )):
            widths = [layer.width for layer in dispatch]
            if index == 0:
                widths = [8] + widths
            else:
                widths = [self.sym_layers[-1].width] + widths
            for n_in, n_out in zip(widths[:-1], widths[1:]):
                key, w_key, b_key = split(key, 3)
                layers.append(LayerParams(**{
                    'weights': scale * normal(w_key, (n_in, n_out)),
                    'biases': scale * normal(b_key, (n_out,)),
                }))
        w_key, b_key = split(key)
        asym_layers.append(LayerParams(**{
            'weights': scale * normal(w_key, (self.asym_layers[-1].width, 1)),
            'biases': scale * normal(b_key, (1,))
        }))
        return NetworkParams(**{
            'sym_layers': sym_layers,
            'asym_layers': asym_layers,
            'activations': [
                lay.activation for lay in self.sym_layers + self.asym_layers
            ]
        })


@partial(jit, static_argnums=2)
@partial(vmap, in_axes=(None, None, None, 0))
def _forward(sym_layers, asym_layers, activations, x):
    x1 = x[0]
    x2 = x[1]
    index = 0
    for layer in sym_layers:
        activation = activation_func(activations[index])
        x1 = activation(x1 @ layer['weights'] + layer['biases'])
        x2 = activation(x2 @ layer['weights'] + layer['biases'])
        index += 1
    x = x1 + x2
    *asym_layers, last = asym_layers
    for layer in asym_layers:
        activation = activation_func(activations[index])
        x = activation(x @ layer['weights'] + layer['biases'])
        index += 1
    return (relu((x @ last['weights'] + last['biases'])) + 1e-3).squeeze()


@partial(jit, static_argnums=2)
@partial(vmap, in_axes=(None, None, None, 0))
def _predict(sym_layers, asym_layers, activations, x):
    x = _large_to_medium(x)
    return _forward(sym_layers, asym_layers, activations, x)
