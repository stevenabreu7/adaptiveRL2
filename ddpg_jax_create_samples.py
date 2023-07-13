import jax
import gymnasium as gym
import numpy as np
import random
import time
from tqdm import tqdm
from ddpg_utils import ReplayBuffer


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
