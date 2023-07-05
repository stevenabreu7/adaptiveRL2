import jax
import jax.numpy as jnp
import gymnasium as gym
import flax
import flax.linen as nn
import numpy as np
import optax
import os
import random
import time
from collections import namedtuple, deque
from flax.training.train_state import TrainState
from tqdm import tqdm
from typing import Sequence


Experience = namedtuple("experience", "state action reward next_state done")

class Logger:
    def __init__(self, log_folder, update_freq=10) -> None:
        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok=True)
        self.datastore = {}
        self.counter = {}
        self.update_freq = update_freq
    
    def write_array(self, arr: jnp.array, filename: str, idx: int):
        jnp.save(f'./{self.log_folder}/{filename}_{idx}.npy', arr)
    
    def write_scalar(self, scalar: float, filename: str, idx: int, update_freq=None):
        self.datastore[filename] = self.datastore.get(filename, []) + [scalar]
        # update every self.update_freq steps
        self.counter[filename] = self.counter.get(filename, 0) + 1
        ufreq = update_freq if update_freq is not None else self.update_freq
        if self.counter[filename] >= ufreq:
            self.save()
            self.counter[filename] = 0

    def save(self):
        for filename, data in self.datastore.items():
            jnp.save(f'./{self.log_folder}/{filename}.npy', np.array(data))

    def close(self):
        self.save()

class ReplayBuffer:
	"""Replay buffer to store and sample experience tuples."""

	def __init__(self, buffer_size: int):
		self.memory = deque(maxlen=buffer_size)

	def add(self, state, action, reward, next_state, done):
		e = Experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self, batch_size: int) -> Sequence[jnp.ndarray]:
		"""Randomly sample a batch of experiences from memory."""
        # NOTE: do we have to do something about the random seed here?
		s, a, r, n, d = zip(*random.sample(self.memory, k=batch_size))
		return (
			jnp.vstack(s, dtype=float), jnp.vstack(a, dtype=int), jnp.vstack(r, dtype=float),
			jnp.vstack(n, dtype=float), jnp.vstack(d, dtype=float)
		)

	def __len__(self):
		return len(self.memory)

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
    action_dim: Sequence[int]
    action_scale: Sequence[int]
    action_bias: Sequence[int]

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

class DDPGTrainState(TrainState):
    target_params: flax.core.FrozenDict

@jax.jit
def update_critic(actor_state: DDPGTrainState, qf_state: DDPGTrainState, states: np.ndarray,
                  actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
                  dones: np.ndarray, gamma: float):
    # NOTE: changed actor.apply -> actor_state.apply_fn
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
    buffer_size = 10_000
    learning_rate = 1e-3 # 3e-4
    batch_size = 128 # 256
    # discount factor
    gamma = 0.99
    # frequency of training the policy (delayed)
    policy_frequency = 2
    # target smoothing coefficient
    tau = 0.005
    # number of initial random steps
    learning_starts = 1_000 # 25_000
    # scale of exploration noise
    exploration_noise = 0.1

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
    replay_buffer = ReplayBuffer(buffer_size=buffer_size)

    # setup networks
    scale_action = jnp.array((env.action_space.high - env.action_space.low) / 2.)
    bias_action = jnp.array((env.action_space.high + env.action_space.low) / 2.)
    actor = Actor(
        action_dim=dim_action,
        action_scale=scale_action,
        action_bias=bias_action
    )
    qf = QNetwork()

    # jit apply functions for speed
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    # setup (custom) training states
    actor_state = DDPGTrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, sample_state),
        target_params=actor.init(actor_key, sample_state),
        tx=optax.adam(learning_rate=learning_rate)
    )
    qf_state = DDPGTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf_key, sample_state, sample_action),
        target_params=qf.init(qf_key, sample_state, sample_action),
        tx=optax.adam(learning_rate=learning_rate)
    )

    global_step = 0
    for episode in tqdm(range(n_episodes)):
        ep_start_time = time.time()
        state, _ = env.reset()
        ep_reward = 0

        for timestep in range(max_timesteps):
              
            if global_step < learning_starts:
                # if not learning, sample random action (exploration)
                action = env.action_space.sample()
            else:
                # if learning, sample action from actor, with noise
                action = actor.apply(actor_state.params, state)
                action_noise = jax.random.normal(key) * scale_action * exploration_noise
                action = (jax.device_get(action) + action_noise).clip(
                    env.action_space.low, env.action_space.high
                )

            # step the environment
            next_state, reward, done, trunc, info = env.step(action)
            ep_reward += reward

            # TODO should we handle trunc first?
            replay_buffer.add(state, action, reward, next_state, done)
            # update state (don't use the real next state?)
            state = next_state

            if global_step > learning_starts:
                # do learning step: sample, update critic, update actor (every N)
                s, a, r, n, d = replay_buffer.sample(batch_size)
                qf_state, qf_loss, qf_a = update_critic(
                    actor_state, qf_state,
                    s, a, n, r, d, gamma
                )
                if global_step % policy_frequency == 0:
                    actor_state, qf_state, actor_loss = update_actor(
                        actor_state, qf_state, s, tau
                    )
                # store logs
                if global_step % 100 == 0:
                    ep_dur = time.time() - ep_start_time
                    logger.write_scalar(scalar=qf_loss, filename='qf_loss', idx=global_step)
                    logger.write_scalar(scalar=actor_loss, filename='actor_loss', idx=global_step)
                    logger.write_scalar(scalar=qf_a, filename='qf_a_values', idx=global_step)
                    logger.write_scalar(scalar=ep_dur, filename='ep_dur', idx=global_step)

            # TODO handle truncated better? for now just reset env
            if trunc or done:
                break
            # handle terminal_observation - check if truncated
            # real next state is used to train the agent
            # real_next_state = next_state.copy()
            # for idx, d in enumerate(trunc):
                # if d:
                    # jax.debug.breakpoint()
                    # real_next_state[idx] = info['final_observation'][idx]
            # TODO should we reset the env here?
            # TODO: what does truncated do in this env?
            # if trunc:
            #     next_state, _ = env.reset()
            #     print('truncated')

            global_step += 1

        # store number of timesteps
        logger.write_scalar(scalar=timestep, filename='ep_timesteps', idx=episode)
        logger.write_scalar(scalar=ep_reward, filename='ep_reward', idx=episode)

    # total_timesteps = 1_000_000
    # for global_step in range(total_timesteps):
    #     if global_step < learning_starts:
    #         # if not learning, sample random action (exploration)
    #         action = env.action_space.sample()
    #     else:
    #         # if learning, sample action from actor, with noise
    #         action = actor.apply(actor_state.params, state)
    #         action_noise = jax.random.normal(key) * scale_action * exploration_noise
    #         action = (jax.device_get(action) + action_noise).clip(
    #             env.action_space.low, env.action_space.high
    #         )

    #     next_state, reward, done, trunc, info = env.step(action)
    #     logger.write_scalar(reward, 'reward', global_step, update_freq=100)

    #     # record rewards
    #     if "final_info" in info:
    #         for info in info["final_info"]:
    #             print('global_step={}, ep_ret={}, ep_len={}'.format(
    #                 global_step, info["episode"]["r"], info["episode"]["l"]
    #             ))
    #             break
        
    #     # handle terminal_observation - check if truncated
    #     # real next state is used to train the agent
    #     # real_next_state = next_state.copy()
    #     # for idx, d in enumerate(trunc):
    #         # if d:
    #             # jax.debug.breakpoint()
    #             # real_next_state[idx] = info['final_observation'][idx]

    #     # TODO should we reset the env here?
    #     # TODO: what does truncated do in this env?
    #     if trunc:
    #         next_state, _ = env.reset()
    #         print('truncated')

    #     # TODO should we store the real next state here? or should we reset env?
    #     replay_buffer.add(state, action, reward, next_state, done)
    #     # update state (don't use the real next state?)
    #     state = next_state

    #     if global_step > learning_starts:
    #         s, a, r, n, d = replay_buffer.sample(batch_size)
    #         qf_state, qf_loss, qf_a = update_critic(
    #             actor_state, qf_state,
    #             s, a, n, r, d, gamma
    #         )
    #         if global_step % policy_frequency == 0:
    #             actor_state, qf_state, actor_loss = update_actor(
    #                 actor_state, qf_state, s, tau
    #             )
    #         if global_step % 100 == 0:
    #             sps = int(global_step / (time.time() - start_time))
    #             print('seconds/step:', sps)
    #             logger.write_scalar(scalar=qf_loss, filename='qf_loss', idx=global_step)
    #             logger.write_scalar(scalar=actor_loss, filename='actor_loss', idx=global_step)
    #             logger.write_scalar(scalar=qf_a, filename='qf_a_values', idx=global_step)
    #             logger.write_scalar(scalar=sps, filename='sps', idx=global_step)
