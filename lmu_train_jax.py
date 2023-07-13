import dataclasses
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.training.train_state as ts
import optax
import torch.utils.data as td
import tensorboard

import data
from models.control import LmuMlpWithAction
from ddpg_jax import Logger


@dataclasses.dataclass
class Config:
    batch_size = 16
    learning_rate: float = 1e-3
    momentum: float = 1e-6
    lmu_theta: int = 4
    lmu_hidden: int = 8
    lmu_memory: int = 12
    lmu_dim_out: int = 4


# @jax.jit
def apply_model(state, xs, ys, lengths):
    def batch_loss(params):
        def loss_fn(xs, ys, l):
            lmu_state = None
            preds = []
            jax.debug.print("l {l}, {xs}, {ys}", l=l, xs=xs.shape, ys=ys.shape)
            for idx in range(l):  # Loop over time
                x0 = xs[idx]
                # jax.debug.print("idx {i}: {x}", i=idx, x=x0)
                new_pred, lmu_state = state.apply_fn(params, x0, lmu_state)
                preds.append(new_pred)
            return (jnp.array(preds[:l]) - ys[:l]) ** 2
            # Remove padded states
            # nonzero_idx = jnp.argwhere(jnp.sum(pred_state, axis=1) > 0)
            # loss = jnp.abs(pred_state[nonzero_idx] - x1[nonzero_idx]) ** 2

        loss = jax.vmap(jax.jit(loss_fn, static_argnums=(2,)), out_axes=(0))(
            xs, ys, lengths
        )
        return jnp.mean(loss)

    return jax.value_and_grad(batch_loss)(state.params)


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng, config):
    network = LmuMlpWithAction(
        4, config.lmu_hidden, config.lmu_memory, config.lmu_theta, config.lmu_dim_out
    )
    params = network.init(rng, jnp.ones([5]))
    tx = optax.adam(config.learning_rate, config.momentum)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=tx)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    env_name = "LMU_pretain"
    logger = Logger(
        log_folder=f'logs/{env_name}/{time.strftime("%m%d_%H%M", time.gmtime())}'
    )
    config = Config()
    writer = tensorboard.summary.Writer("logs")

    train_state = create_train_state(key, config)
    dataset = data.ExploreDataset()
    dataloader = td.DataLoader(dataset, config.batch_size)

    for batch in dataloader:
        (xs, ys, lengths) = batch
        lmu_state = None

        grads, loss = apply_model(
            train_state,
            jnp.array(xs),
            jnp.array(ys),
            jnp.array(lengths, dtype=jnp.float32),
        )
        state = update_model(state, grads)
        writer.scalar("train/loss", loss)

    writer.flush()
