from typing import Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn

from . import lmu_jax

class LmuMlp(nn.Module):
    lmu_input: int
    lmu_q: int
    dense_output: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        state: Optional[jnp.ndarray] = None,
    ):
        x, state = lmu_jax.LMUCell(self.lmu_input, self.lmu_q)(x, state)
        x = nn.Dense(self.dense_output)(x)
        return x, state
