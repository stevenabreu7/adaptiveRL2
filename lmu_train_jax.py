import argparse
import dataclasses
import datetime
import tqdm

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import flax.training.train_state as ts
import flax.training.orbax_utils as ou
import optax
import orbax
from orbax.checkpoint import CheckpointManagerOptions, Checkpointer, CheckpointManager
from flax.training.orbax_utils import restore_args_from_target
import torch.utils.data as td
from tensorboardX import SummaryWriter

import data
from models.control import LmuMlpWithAction, LMUConfig as Config
from ddpg_jax import Logger


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
        return jnp.mean(loss)

    loss, grad = jax.value_and_grad(batch_loss)(state.params)
    return loss, grad


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng, config, with_velocities=True):
    lmu_input = 4 if with_velocities else 2
    network = LmuMlpWithAction(
        lmu_input,
        config.lmu_hidden,
        config.lmu_memory,
        config.lmu_theta,
        config.lmu_dim_out,
        learn_a=True,
        learn_b=True,
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

        loss, grads = apply_model(
            state,
            jnp.array(xs),
            jnp.array(ys),
            jnp.array(lengths, dtype=jnp.float32),
        )
        state = update_model(state, grads)
        writer.add_scalar("train/loss", loss, (epoch + 1) * idx)
        plot_grad(grads, writer, (epoch + 1) * idx)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for LMU autoencoder")
    parser.add_argument("--no-velocities", dest="velocities", action="store_false")
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--restore", type=str, default=None)
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    config = Config(with_velocities=args.velocities)
    timestamp = datetime.datetime.today().strftime("%y%m%d_%H%M")
    writer = SummaryWriter(f"logs/{timestamp}")
    writer.add_hparams(dataclasses.asdict(config), {})

    # create checkpoint manager
    chpt_dir = (
        f"checkpoints/{timestamp}"
        if args.restore is None
        else args.restore
    )
    mgr_options = CheckpointManagerOptions(
        create=True, max_to_keep=3, keep_period=2, step_prefix="train"
    )
    chptr = Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ckpt_mgr = CheckpointManager(chpt_dir, chptr, mgr_options)

    # create training state and data loaders
    key = jax.random.PRNGKey(0)
    config = Config()
    train_state = create_train_state(key, config)
    dataset = data.ExploreDataset(args.data_path)
    dataloader = td.DataLoader(dataset, config.batch_size)

    # load training state from checkpoint
    if args.restore is not None:
        step = ckpt_mgr.latest_step()
        step = 0 if step is None else step
        ckpt_path = f"{chpt_dir}/train_{step}/default"
        empty_state = {"model": train_state, "config": dataclasses.asdict(config)}
        train_state = chptr.restore(ckpt_path, item=empty_state)

    for epoch in tqdm.tqdm(range(100)):
        train_state = train_epoch(dataloader, train_state, writer, epoch)
        ckpt = {"model": train_state, "config": dataclasses.asdict(config)}
        save_args = flax.training.orbax_utils.save_args_from_target(ckpt)
        ckpt_mgr.save(epoch, ckpt, save_kwargs={"save_args": save_args})

    writer.flush()
