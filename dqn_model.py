import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence

class MLP(nn.Module):
  """Simple MLP module."""
  architecture: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.architecture[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.architecture[-1])(x)
    return x

def init_mlp_key(key, architecture):
    """Initialize MLP with given architecture and random key.
    
    Args:
        key: random key
        architecture (list(int)): number of neurons per layer in MLP
    Returns:
        model (nn.Module): MLP model
        params (pytree): MLP parameters
    """
    model = MLP(architecture)
    batch = jnp.ones((7, architecture[0]))
    params = model.init(key, batch)
    return model, params

def init_mlp_seed(seed, architecture):
    """Initialize MLP with given architecture and random seed.

    Args:
        seed (int): random seed
        architecture (list(int)): number of neurons per layer in MLP
    Returns:
        model (nn.Module): MLP model
        params (pytree): MLP parameters
    """
    model = MLP(architecture)
    batch = jnp.ones((7, architecture[0]))
    params = model.init(jax.random.PRNGKey(seed), batch)
    return model, params
