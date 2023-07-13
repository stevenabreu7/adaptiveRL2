# TODO: check if LMU cell is fixed or trained
#   how to train it easily?
import jax
import jax.numpy as jnp
import gymnasium as gym
import numpy as np
import optax
import random
import time
import flax.linen as nn
from typing import Sequence
from tqdm import tqdm
from models.lmu_jax import LMUCell
from flax.training.train_state import TrainState
from ddpg_utils import Logger, DDPGTrainState, Experience, TemporalReplayBuffer


class QNetwork(nn.Module):
    """Q Network: (state, action) -> Q-value."""
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    """Actor Network: state -> action."""
    action_dim: int
    action_scale: Sequence[float]
    action_bias: Sequence[float]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


@jax.jit
def update_critic(actor_state: DDPGTrainState, qf_state: DDPGTrainState, states: np.ndarray,
                  actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
                  dones: np.ndarray, gamma: float):
    next_state_actions = actor_state.apply_fn(actor_state.target_params, next_states)
    next_state_actions = next_state_actions.clip(-1, 1)  # TODO: proper clip
    qf_next_target = qf_state.apply_fn(qf_state.target_params, next_states, 
                                       next_state_actions).reshape(-1)
    next_q_value = (rewards.squeeze(1) + (1 - dones.squeeze(1)) * gamma * (qf_next_target)).reshape(-1)

    def mse_loss(params):
        qf_a_values = qf_state.apply_fn(params, states, actions).squeeze()
        return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

    (qf_loss, qf_a), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params)
    qf_state = qf_state.apply_gradients(grads=grads)
    return qf_state, qf_loss, qf_a

@jax.jit
def update_actor(actor_state: DDPGTrainState, qf_state: DDPGTrainState, states: np.ndarray,
                 tau: float):
    # tau: step size for the optimizer
    def actor_loss(params):
        loss = -qf_state.apply_fn(
             qf_state.params, states, actor.apply(params, states)
        ).mean()
        return loss

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    actor_state = actor_state.replace(
        target_params=optax.incremental_update(actor_state.params, actor_state.target_params, tau)
    )
    qf_state = qf_state.replace(
        target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau)
    )
    return actor_state, qf_state, actor_loss_value

if __name__ == "__main__":
    seed = 0
    n_episodes = 1_000
    max_timesteps = 1_000
    buffer_size = 50_000
    learning_rate = 1e-3 # 3e-4
    batch_size = 128 # 256
    # discount factor
    gamma = 0.99
    # frequency of training the policy (delayed)
    policy_frequency = 2
    # target smoothing coefficient
    tau = 0.005
    # number of initial random steps
    learning_starts = 10_000 # 25_000 * 6 # 25_000  # avg 6 random steps / episode
    # scale of exploration noise
    exploration_noise = 1e-3

    # parameters for LMU to encode the state
    lmu_memory_size = 16
    lmu_hidden_size = 4
    lmu_theta = 8
    lmu_learn_a = False
    lmu_learn_b = False
    # lmu_decay = 0.5
    # lmu_dt = 1.0

    # env_name = 'MountainCarContinuous-v0'
    env_name = 'InvertedPendulum-v4'

    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)

    logger = Logger(log_folder=f'logs/{env_name}/{time.strftime("%m%d_%H%M", time.gmtime())}')

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
    replay_buffer = TemporalReplayBuffer(buffer_size=buffer_size)

    # setup networks
    scale_action = jnp.array((env.action_space.high - env.action_space.low) / 2.)
    bias_action = jnp.array((env.action_space.high + env.action_space.low) / 2.)
    actor = Actor(
        action_dim=dim_action,
        action_scale=scale_action,
        action_bias=bias_action
    )
    qf = QNetwork()
    lmu = LMUCell(input_size=dim_state, hidden_size=lmu_hidden_size, memory_size=lmu_memory_size, 
                  theta=lmu_theta, learn_a=lmu_learn_a, learn_b=lmu_learn_b)

    # jit apply functions for speed
    # actor.apply = jax.jit(actor.apply)
    # qf.apply = jax.jit(qf.apply)
    # # lmu.apply = jax.vmap(jax.jit(lmu.apply))
    # lmu.apply = jax.jit(lmu.apply)

    # setup (custom) training states
    actor_state = DDPGTrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, jnp.zeros(lmu_hidden_size)),
        target_params=actor.init(actor_key, jnp.zeros(lmu_hidden_size)),
        tx=optax.adam(learning_rate=learning_rate)
    )
    qf_state = DDPGTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf_key, jnp.zeros(lmu_hidden_size), sample_action),
        target_params=qf.init(qf_key, jnp.zeros(lmu_hidden_size), sample_action),
        tx=optax.adam(learning_rate=learning_rate)
    )
    lmu_state = TrainState.create(
        apply_fn=lmu.apply,
        params=lmu.init(key, jnp.ones((1, dim_state))),
        tx=optax.adam(learning_rate=learning_rate)
    )

    print('exploration phase')
    ep_reward = 0
    last_ep_end = 0
    state, _ = env.reset()
    episode_experiences = []
    for step in tqdm(range(learning_starts)):
        # sample random action
        action = env.action_space.sample()
        # step the environment
        next_state, reward, done, trunc, info = env.step(action)
        ep_reward += reward
        # save experience for the temporal replay buffer later
        episode_experiences.append(Experience(state, action, reward, next_state, done))
        # update state
        state = next_state
        if done or trunc: 
            # move to next episode (store episode statistics)
            state, _ = env.reset()
            logger.write_scalar(scalar=step-last_ep_end, filename='ep_timesteps', idx=0)
            logger.write_scalar(scalar=ep_reward, filename='ep_reward', idx=0)
            ep_reward = 0
            last_ep_end = step
            # add episode experiences to replay buffer
            replay_buffer.add(episode_experiences)
            episode_experiences = []

    print('training phase')
    with jax.profiler.trace("tmp/jaxtrace"):
        iterator = tqdm(range(n_episodes))

        for episode in iterator:
            ep_start_time = time.time()
            state, _ = env.reset()
            ep_reward = 0
            ep_experiences = []
            lmu_int_state = None

            for timestep in range(max_timesteps):
                # compute LMU encoding
                lmu_int_state = lmu.apply(lmu_state.params, jnp.array(state), lmu_int_state)
                lmu_out, _ = lmu_int_state

                # sample action from actor, with noise
                # action = actor.apply(actor_state.params, jnp.concatenate([lmu_out, state]))
                action = actor.apply(actor_state.params, lmu_out)
                action_noise = jax.random.normal(key) * scale_action * exploration_noise
                action = (jax.device_get(action) + action_noise).clip(
                    env.action_space.low, env.action_space.high
                )

                # step the environment
                next_state, reward, done, trunc, info = env.step(action)
                ep_reward += reward

                # store experience for the temporal replay buffer later
                ep_experiences.append(Experience(state, action, reward, next_state, done))

                # update state (don't use the real next state?)
                state = next_state

                # TODO handle truncated better? for now just reset env
                if trunc or done:
                    break
            
            # add episode experiences to replay buffer
            replay_buffer.add(ep_experiences)
            ep_experiences = []

            # do learning step: sample, update critic, update actor (every N)
            s, a, r, n, d = replay_buffer.sample(batch_size)
            lmu_int_state = None
            for t in range(len(s)):
                # compute LMU encoding
                lmu_int_state = lmu.apply(lmu_state.params, s[t], lmu_int_state)
                lmu_out, _ = lmu_int_state

                qf_state, qf_loss, qf_a = update_critic(
                    actor_state, qf_state,
                    s[t], a[t], n[t], r[t], d[t], gamma
                )
                if episode % policy_frequency == 0:
                    actor_state, qf_state, actor_loss = update_actor(
                        actor_state, qf_state, s[t], tau
                    )

            # store logs
            ep_dur = time.time() - ep_start_time
            logger.write_scalar(scalar=qf_loss, filename='qf_loss', idx=episode)
            logger.write_scalar(scalar=actor_loss, filename='actor_loss', idx=episode)
            logger.write_scalar(scalar=qf_a, filename='qf_a_values', idx=episode)
            logger.write_scalar(scalar=ep_dur, filename='ep_dur', idx=episode)
            
            # episode ended -> reset LMU
            lmu_params = lmu.init(key, jnp.ones((1, dim_state)))
            lmu_int_state = None

            # store number of timesteps
            logger.write_scalar(scalar=timestep, filename='ep_timesteps', idx=episode)
            logger.write_scalar(scalar=ep_reward, filename='ep_reward', idx=episode)
