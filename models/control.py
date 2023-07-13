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
        x, state = lmu_jax.LMUCellCompact(self.lmu_input, self.lmu_q)(x, state)
        x = nn.Dense(self.dense_output)(x)
        return x, state


class LmuMlpWithAction(nn.Module):
    lmu_input: int
    lmu_hidden: int
    lmu_memory: int
    lmu_theta: int
    dense_output: int
    learn_a: bool = False
    learn_b: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ):
        env_state, action = x[:self.lmu_input], x[4:]
        env_state = lmu_jax.LMUCell(
            self.lmu_input, self.lmu_hidden, self.lmu_memory, self.lmu_theta, self.learn_a, self.learn_b
        )(env_state)
        action_x = jnp.concatenate((env_state, action))
        action_x = nn.Dense(self.dense_output)(action_x)
        return action_x
