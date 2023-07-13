import dataclasses
import datetime
import tqdm

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import flax.training.train_state as ts
import optax
import torch.utils.data as td
from tensorboardX import SummaryWriter

import data
from models.control import LmuMlpWithAction
from ddpg_jax import Logger


@dataclasses.dataclass
class Config:
    batch_size = 16
    learning_rate: float = 1e-3
    momentum: float = 1e-6
    lmu_theta: int = 4
    lmu_hidden: int = 4
    lmu_memory: int = 12
    lmu_dim_out: int = 4

@jax.jit
def apply_model(state, xs, ys, lengths):
    def batch_loss(params):
        def loss_fn(xs, ys, l):
            lmu_state = None
            losses = []
            # length = int(l)
            # jax.debug.print("l {l}({l1}), {xs}, {ys}", l=l, xs=xs.shape, ys=ys.shape, l1=length)
            # indices = 
            # Remove padded states
            # nonzero_idx = jnp.argwhere(jnp.sum(xs, axis=1) > 0)
            # xs = xs[nonzero_idx]
            # ys = ys[nonzero_idx]
            for x, y in zip(xs, ys):  # Loop over time
                # jax.debug.print("idx {i}: {x}", i=idx, x=x0)
                if lmu_state is None:
                    states = {"params": params}
                else:
                    states = {"params": params, "state": lmu_state}
                pred, lmu_state = state.apply_fn(states, x, mutable=["state"])
                loss_step = jnp.mean((pred - y) ** 2)
                losses.append(loss_step)
            return jnp.array(losses)

            # Remove padded states
            # nonzero_idx = jnp.argwhere(jnp.sum(pred_state, axis=1) > 0)
            # loss = jnp.abs(pred_state[nonzero_idx] - x1[nonzero_idx]) ** 2
        loss = jax.vmap(loss_fn, out_axes=(0))(xs, ys, lengths)
        # jax.debug.breakpoint()
        return jnp.mean(loss)

    loss, grad = jax.value_and_grad(batch_loss)(state.params)
    return loss, grad


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng, config):
    network = LmuMlpWithAction(
        4, config.lmu_hidden, config.lmu_memory, config.lmu_theta, config.lmu_dim_out,
        learn_a = True, learn_b = True
    )
    variables = network.init(rng, jnp.ones([5]))
    state, params = flax.core.pop(variables, "params")
    tx = optax.adam(config.learning_rate, config.momentum)
    return ts.TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def plot_grad(d, writer, step):
    for k, v in d.items():
        if isinstance(v, flax.core.frozen_dict.FrozenDict):
            plot_grad(v, writer, step)
        else:
            writer.add_histogram(f"train/grad/{k}", v, step)

def train_epoch(dataloader, state, writer, epoch):
    for idx, batch in tqdm.tqdm(enumerate(dataloader), leave=False):
        (xs, ys, lengths) = batch
        lmu_state = None

        loss, grads = apply_model(
            state,
            jnp.array(xs),
            jnp.array(ys),
            jnp.array(lengths, dtype=jnp.float32),
        )
        state = update_model(state, grads)
        writer.add_scalar("train/loss", loss, (epoch + 1) * idx)
        plot_grad(grads, writer, (epoch + 1) * idx)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    config = Config()
    timestamp = datetime.datetime.today().strftime("%y%m%d_%H%M")
    writer = SummaryWriter(f"logs/{timestamp}")

    train_state = create_train_state(key, config)
    dataset = data.ExploreDataset()
    dataloader = td.DataLoader(dataset, config.batch_size)

    for epoch in tqdm.tqdm(range(100)):
        train_epoch(dataloader, train_state, writer, epoch)
    writer.flush()
