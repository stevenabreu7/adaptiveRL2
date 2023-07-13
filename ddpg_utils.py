import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import os
import random
from flax.training.train_state import TrainState
from collections import namedtuple, deque
from typing import Sequence


Experience = namedtuple("experience", "state action reward next_state done")
ExperienceLMU = namedtuple("experience", "state action reward next_state done lmu_state lmu_next_state")


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

	def save_to_file(self, filename: str) -> True:
		s, a, r, n, d = zip(*list(self.memory))
		s, a, r, n, d = (
			jnp.vstack(s, dtype=float), jnp.vstack(a, dtype=float), jnp.vstack(r, dtype=float),
			jnp.vstack(n, dtype=float), jnp.vstack(d, dtype=float)
		)
		np.savez(filename, s=s, a=a, r=r, n=n, d=d)
		return True
	
	def load_from_file(self, filename: str) -> bool:
		x = np.load(filename)
		s, a, r, n, d = x['s'], x['a'], x['r'], x['n'], x['d']
		for i in range(len(s)):
			self.add(s[i], a[i], r[i], n[i], d[i])
		return True


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
        states = np.zeros((batch_size, max_len, exp.state.shape[0]))
        actions = np.zeros((batch_size, max_len, exp.action.shape[0]))
        rewards = np.zeros((batch_size, max_len, 1))
        next_states = np.zeros((batch_size, max_len, exp.next_state.shape[0]))
        dones = np.zeros((batch_size, max_len, 1))
        for idx, experiences in enumerate(experiences_list):
            states[idx, :len(experiences)] = jnp.vstack([e.state for e in experiences])
            actions[idx, :len(experiences)] = jnp.vstack([e.action for e in experiences])
            rewards[idx, :len(experiences)] = jnp.vstack([e.reward for e in experiences])
            next_states[idx, :len(experiences)] = jnp.vstack([e.next_state for e in experiences])
            dones[idx, :len(experiences)] = jnp.vstack([e.done for e in experiences])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class ReplayBufferLMU:
	"""Replay buffer to store and sample experience tuples."""

	def __init__(self, buffer_size: int):
		self.memory = deque(maxlen=buffer_size)

	def add(self, state, action, reward, next_state, done, lmu_state, lmu_next_state):
		e = ExperienceLMU(state, action, reward, next_state, done, lmu_state, lmu_next_state)
		self.memory.append(e)

	def sample(self, batch_size: int) -> Sequence[jnp.ndarray]:
		"""Randomly sample a batch of experiences from memory."""
        # NOTE: do we have to do something about the random seed here?
		s, a, r, n, d, l, ln = zip(*random.sample(self.memory, k=batch_size))
		return (
			jnp.vstack(s, dtype=float), jnp.vstack(a, dtype=int), jnp.vstack(r, dtype=float),
			jnp.vstack(n, dtype=float), jnp.vstack(d, dtype=float), jnp.vstack(l, dtype=float), 
            jnp.vstack(ln, dtype=float)
		)

	def __len__(self):
		return len(self.memory)


class DDPGTrainState(TrainState):
    target_params: flax.core.FrozenDict
