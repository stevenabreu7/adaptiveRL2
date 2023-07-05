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
        outputs = []
        cell = lmu_jax.LMUCell(self.lmu_input, self.lmu_q)
        linear = nn.Dense(self.lmu_input * self.lmu_q, self.dense_output)
        x, state = cell(x, state)
        x = linear(x)
        return outputs, state
