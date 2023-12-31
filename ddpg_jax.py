# TODO: make LMU cell fixed (not trainable)
import jax
import jax.numpy as jnp
import gymnasium as gym
import numpy as np
import optax
import random
import time
from tqdm import tqdm
from models.lmu_jax import LMUCell
from collections import deque
from typing import Sequence
from ddpg_utils import Logger, ReplayBufferLMU, DDPGTrainState, Experience


class TemporalReplayBuffer:
    """Replay buffer to store and sample experience tuples."""

    def __init__(self, buffer_size: int):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experiences: Sequence[Experience]):
        self.memory.append(experiences)

    def sample(self, batch_size: int) -> Sequence[jnp.ndarray]:
        """Randomly sample a batch of experience sequences from memory."""
        experiences_list = random.sample(self.memory, k=batch_size)
        max_len = max([len(experiences) for experiences in experiences_list])
        exp = experiences_list[0][0]
        states = jnp.zeros((batch_size, max_len, exp.state.shape[0]))
        actions = jnp.zeros((batch_size, max_len, exp.action.shape[0]))
        rewards = jnp.zeros((batch_size, max_len, exp.reward.shape[0]))
        next_states = jnp.zeros((batch_size, max_len, exp.next_state.shape[0]))
        dones = jnp.zeros((batch_size, max_len, exp.done.shape[0]))
        for idx, experiences in enumerate(experiences_list):
            states[idx, :len(experiences)] = jnp.vstack([e.state for e in experiences])
            actions[idx, :len(experiences)] = jnp.vstack([e.action for e in experiences])
            rewards[idx, :len(experiences)] = jnp.vstack([e.reward for e in experiences])
            next_states[idx, :len(experiences)] = jnp.vstack([e.next_state for e in experiences])
            dones[idx, :len(experiences)] = jnp.vstack([e.done for e in experiences])
        return states, actions, rewards, next_states, dones


@jax.jit
def update_critic(
    actor_state: DDPGTrainState,
    qf_state: DDPGTrainState,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    lmu_states: np.ndarray,
    lmu_next_states: np.ndarray,
    gamma: float,
):
    # NOTE: changed actor.apply -> actor_state.apply_fn
    merged_states = merge_env_state_lmu_state(states, lmu_states)
    merged_next_states = merge_env_state_lmu_state(next_states, lmu_next_states)
    next_state_actions = actor_state.apply_fn(
        actor_state.target_params, merged_next_states
    )
    next_state_actions = next_state_actions.clip(-1, 1)  # TODO: proper clip
    qf_next_target = qf_state.apply_fn(
        qf_state.target_params, merged_next_states, next_state_actions
    ).reshape(-1)
    next_q_value = (
        rewards.squeeze(1) + (1 - dones.squeeze(1)) * gamma * (qf_next_target)
    ).reshape(-1)

    def mse_loss(params):
        qf_a_values = qf_state.apply_fn(params, merged_states, actions).squeeze()
        return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

    (qf_loss, qf_a), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params)
    qf_state = qf_state.apply_gradients(grads=grads)
    return qf_state, qf_loss, qf_a


@jax.jit
def update_actor(
    actor_state: DDPGTrainState,
    qf_state: DDPGTrainState,
    states: np.ndarray,
    lmu_states: np.ndarray,
    tau: float,
):
    # tau: step size for the optimizer
    def actor_loss(params):
        merged_states = merge_env_state_lmu_state(states, lmu_states)
        loss = -qf_state.apply_fn(
            qf_state.params, merged_states, actor.apply(params, merged_states)
        ).mean()
        return loss

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    actor_state = actor_state.replace(
        target_params=optax.incremental_update(
            actor_state.params, actor_state.target_params, tau
        )
    )
    qf_state = qf_state.replace(
        target_params=optax.incremental_update(
            qf_state.params, qf_state.target_params, tau
        )
    )
    return actor_state, qf_state, actor_loss_value


if __name__ == "__main__":
    seed = 0
    n_episodes = 1_000
    max_timesteps = 1_000
    buffer_size = 100_000
    learning_rate = 1e-3 # 3e-4
    batch_size = 128 # 256
    # discount factor
    gamma = 0.99
    # frequency of training the policy (delayed)
    policy_frequency = 2
    # target smoothing coefficient
    tau = 0.005
    # number of initial random steps
    learning_starts = 25_000 # 25_000
    # scale of exploration noise
    exploration_noise = 1e-3

    # parameters for LMU to encode the state
    lmu_dim_out = 8
    lmu_theta = 4
    lmu_decay = 0.5
    lmu_dt = 1.0

    # env_name = 'MountainCarContinuous-v0'
    env_name = "InvertedPendulum-v4"

    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)

    logger = Logger(
        log_folder=f'logs/{env_name}/{time.strftime("%m%d_%H%M", time.gmtime())}'
    )

    # setup environment
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    sample_state, _ = env.reset()
    sample_action = env.action_space.sample()

    # setup replay buffer
    replay_buffer = ReplayBufferLMU(buffer_size=buffer_size)

    # setup networks
    scale_action = jnp.array((env.action_space.high - env.action_space.low) / 2.0)
    bias_action = jnp.array((env.action_space.high + env.action_space.low) / 2.0)
    actor = Actor(
        action_dim=dim_action, action_scale=scale_action, action_bias=bias_action
    )
    qf = QNetwork()

    # jit apply functions for speed
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    # setup (custom) training states
    actor_state = DDPGTrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, jnp.zeros(dim_state + lmu_dim_out * lmu_theta)),
        target_params=actor.init(
            actor_key, jnp.zeros(dim_state + lmu_dim_out * lmu_theta)
        ),
        tx=optax.adam(learning_rate=learning_rate),
    )
    qf_state = DDPGTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(
            qf_key, jnp.zeros(dim_state + lmu_dim_out * lmu_theta), sample_action
        ),
        target_params=qf.init(
            qf_key, jnp.zeros(dim_state + lmu_dim_out * lmu_theta), sample_action
        ),
        tx=optax.adam(learning_rate=learning_rate),
    )

    print("exploration phase")
    ep_reward = 0
    last_ep_end = 0
    state, _ = env.reset()
    # setup LMU encoder
    lmu_kwargs = {
        "size_in": dim_state,
        "q": lmu_dim_out,
        "theta": lmu_theta,
        "decay": lmu_decay,
        "dt": lmu_dt,
    }
    lmu = LMUCell(**lmu_kwargs)
    lmu_params = lmu.init(key, jnp.ones((1, dim_state)))
    lmu_state = None
    for step in tqdm(range(learning_starts)):
        # compute LMU encoding -> of size (dim_env_state * lmu_dim_out (i.e. q))
        lmu_output, lmu_state = lmu.apply(lmu_params, jnp.array(state), lmu_state)
        # print('LMU output & state:', lmu_output.shape, lmu_state.shape)
        # sample random action
        action = env.action_space.sample()
        # step the environment
        next_state, reward, done, trunc, info = env.step(action)
        ep_reward += reward
        # compute next state LMU encoding
        lmu_next_output, lmu_next_state = lmu.apply(
            lmu_params, jnp.array(next_state), lmu_state
        )
        # store experience to replay buffer
        replay_buffer.add(
            state, action, reward, next_state, done, lmu_output, lmu_next_output
        )
        # update state
        state = next_state
        if done or trunc:
            # move to next episode (store episode statistics)
            state, _ = env.reset()
            logger.write_scalar(
                scalar=step - last_ep_end, filename="ep_timesteps", idx=0
            )
            logger.write_scalar(scalar=ep_reward, filename="ep_reward", idx=0)
            ep_reward = 0
            last_ep_end = step
            # reset LMU
            lmu_params = lmu.init(key, jnp.ones((1, dim_state)))
            lmu_state = None

    with jax.profiler.trace("/tmp/jaxtrace"):
        print("training phase")
        global_step = 0
        # setup LMU encoder
        lmu_kwargs = {
            "size_in": dim_state,
            "q": lmu_dim_out,
            "theta": lmu_theta,
            "decay": lmu_decay,
            "dt": lmu_dt,
        }
        lmu = LMUCell(**lmu_kwargs)
        lmu_params = lmu.init(key, jnp.ones((1, dim_state)))
        lmu_state = None
        for episode in tqdm(range(n_episodes)):
            ep_start_time = time.time()
            state, _ = env.reset()
            ep_reward = 0

            for timestep in range(max_timesteps):
                # compute LMU encoding -> of size (dim_env_state * lmu_dim_out (i.e. q))
                lmu_output, lmu_state = lmu.apply(
                    lmu_params, jnp.array(state), lmu_state
                )
                merged_state = np.concatenate([state, lmu_output])
                # sample action from actor, with noise
                action = actor.apply(actor_state.params, merged_state)
                action_noise = jax.random.normal(key) * scale_action * exploration_noise
                action = (jax.device_get(action) + action_noise).clip(
                    env.action_space.low, env.action_space.high
                )

                # step the environment
                next_state, reward, done, trunc, info = env.step(action)
                ep_reward += reward

                # compute LMU encoding for next state
                lmu_next_output, lmu_next_state = lmu.apply(
                    lmu_params, jnp.array(next_state), lmu_state
                )

                # TODO should we handle trunc first?
                replay_buffer.add(
                    state, action, reward, next_state, done, lmu_output, lmu_next_output
                )
                # update state (don't use the real next state?)
                state = next_state

                # do learning step: sample, update critic, update actor (every N)
                s, a, r, n, d, l, ln = replay_buffer.sample(batch_size)
                qf_state, qf_loss, qf_a = update_critic(
                    actor_state, qf_state, s, a, n, r, d, l, ln, gamma
                )
                if global_step % policy_frequency == 0:
                    actor_state, qf_state, actor_loss = update_actor(
                        actor_state, qf_state, s, l, tau
                    )
                # store logs
                if global_step % 100 == 0:
                    ep_dur = time.time() - ep_start_time
                    logger.write_scalar(
                        scalar=qf_loss, filename="qf_loss", idx=global_step
                    )
                    logger.write_scalar(
                        scalar=actor_loss, filename="actor_loss", idx=global_step
                    )
                    logger.write_scalar(
                        scalar=qf_a, filename="qf_a_values", idx=global_step
                    )
                    logger.write_scalar(
                        scalar=ep_dur, filename="ep_dur", idx=global_step
                    )

                # TODO handle truncated better? for now just reset env
                if trunc or done:
                    break

                global_step += 1

            # episode ended -> reset LMU
            lmu_params = lmu.init(key, jnp.ones((1, dim_state)))
            lmu_state = None

            # store number of timesteps
            logger.write_scalar(scalar=timestep, filename="ep_timesteps", idx=episode)
            logger.write_scalar(scalar=ep_reward, filename="ep_reward", idx=episode)
