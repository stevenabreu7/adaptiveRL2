import dataclasses
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.training.train_state as ts
import optax
import torch.utils.data as td

import data
from models.control import LmuMlpWithAction
from ddpg_jax import Logger

@dataclasses.dataclass
class Config():
    batch_size = 128
    learning_rate: float = 1e-3
    momentum: float = 1e-6
    lmu_theta: int = 4
    lmu_dim_out: int = 8


@jax.jit
def apply_model(state, x0, a, s, x1, length):
    print(state, x0, a, s, x1)
    pred_state, new_state = state.apply_fn(x0, a, s)
    # Remove padded states
    # nonzero_idx = jnp.argwhere(jnp.sum(pred_state, axis=1) > 0)
    # loss = jnp.abs(pred_state[nonzero_idx] - x1[nonzero_idx]) ** 2
    loss = jnp.abs(pred_state[:length] - x1[:length]) ** 2
    return loss, new_state

def create_train_state(rng, config):
    network = LmuMlpWithAction(4, config.lmu_theta, config.lmu_dim_out)
    params = network.init(rng, jnp.ones([5]))
    tx = optax.adam(config.learning_rate, config.momentum)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=tx)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    env_name = "LMU_pretain"
    logger = Logger(log_folder=f'logs/{env_name}/{time.strftime("%m%d_%H%M", time.gmtime())}')
    config = Config(lmu_theta = 4)

    # ... Load data
    dataset = None

    train_state = create_train_state(key, config)
    dataset = data.ExploreDataset()
    dataloader = td.DataLoader(dataset, config.batch_size)

    for batch in dataloader:
      (xs, ys, lenghts) = batch
      lmu_state = None
      print(xs.shape)
        
    # X: sequence, 5

    
        # x0 = ... # State 0
        # x1 = ... # State 1
        # a0 = ... # Action 0

        # loss, lmu_state = network(x0, a0, lmu_state)

