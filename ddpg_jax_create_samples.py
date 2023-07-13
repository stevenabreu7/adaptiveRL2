# TODO: make LMU cell fixed (not trainable)
import jax
import jax.numpy as jnp
import gymnasium as gym
import numpy as np
import random
import time
from collections import namedtuple, deque
from tqdm import tqdm
from typing import Sequence


Experience = namedtuple("experience", "state action reward next_state done")


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
            jnp.vstack(s, dtype=float),
            jnp.vstack(a, dtype=float),
            jnp.vstack(r, dtype=float),
            jnp.vstack(n, dtype=float),
            jnp.vstack(d, dtype=float),
        )

    def __len__(self):
        return len(self.memory)

    def save_to_file(self, filename: str) -> True:
        s, a, r, n, d = zip(*list(self.memory))
        s, a, r, n, d = (
            jnp.vstack(s, dtype=float),
            jnp.vstack(a, dtype=float),
            jnp.vstack(r, dtype=float),
            jnp.vstack(n, dtype=float),
            jnp.vstack(d, dtype=float),
        )
        np.savez(filename, s=s, a=a, r=r, n=n, d=d)
        return True

    def load_from_file(self, filename: str) -> bool:
        x = np.load(filename)
        s, a, r, n, d = x["s"], x["a"], x["r"], x["n"], x["d"]
        for i in range(len(s)):
            self.add(s[i], a[i], r[i], n[i], d[i])
        return True


if __name__ == "__main__":
    seed = 0
    buffer_size = 100_000
    learning_starts = buffer_size

    # env_name = 'MountainCarContinuous-v0'
    env_name = "InvertedPendulum-v4"

    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)

    LOG_FOLDER = f'logs/{env_name}/{time.strftime("%m%d_%H%M", time.gmtime())}'

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

    print("exploration phase")
    ep_reward = 0
    last_ep_end = 0
    state, _ = env.reset()
    lmu_state = None
    for step in tqdm(range(learning_starts)):
        if (step > 0 and step % 1000 == 0) or step == 10:
            replay_buffer.save_to_file(f"{LOG_FOLDER}/replay_buffer")
        # sample random action
        action = env.action_space.sample()
        # step the environment
        next_state, reward, done, trunc, info = env.step(action)
        ep_reward += reward
        # store experience to replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        # update state
        state = next_state
        if done or trunc:
            # move to next episode (store episode statistics)
            state, _ = env.reset()
            ep_reward = 0
            last_ep_end = step
