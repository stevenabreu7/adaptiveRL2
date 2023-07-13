import jax
import jax.numpy as jnp
import rlax
import random

from collections import namedtuple, deque
from jax import jit, vmap


Params = namedtuple("params", "policy target")
Experience = namedtuple("experience", "state action reward next_state done")
ExpCfg = namedtuple(
    "exp_cfg",
    "env_name architecture n_episodes n_steps batch_size replay_size learning_rate target_update_frequency gamma eps_schedule",
)

ExperimentConfig = namedtuple("exp_cfg", "env_name architecture")
TrainingConfig = namedtuple(
    "train_cfg",
    "n_episodes n_steps batch_size replay_size learning_rate target_update_frequency gamma eps_schedule",
)


class ReplayBuffer:
    """Replay buffer to store and sample experience tuples."""

    def __init__(self, buffer_size, seed):
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        s, a, r, n, d = zip(*random.sample(self.memory, k=batch_size))
        return (
            jnp.vstack(s, dtype=float),
            jnp.vstack(a, dtype=int),
            jnp.vstack(r, dtype=float),
            jnp.vstack(n, dtype=float),
            jnp.vstack(d, dtype=float),
        )

    def __len__(self):
        return len(self.memory)


@jit
def policy(key, state, trainstate, epsilon, greedy=False):
    """Epsilon-greedy policy. Maps state to action."""
    state = jnp.expand_dims(state, axis=0)
    q = jnp.squeeze(trainstate.apply_fn(trainstate.params, state))
    a_eps = rlax.epsilon_greedy(epsilon).sample(key, q)
    a_grd = rlax.greedy().sample(key, q)
    action = jax.lax.select(greedy, a_grd, a_eps)
    return action


@vmap
def q_learning_loss(q, target_q, action, reward, done, gamma):
    """Compute q-learning loss through TD-learning."""
    td_target = reward + gamma * target_q.max() * (1.0 - done)
    td_error = jax.lax.stop_gradient(td_target) - q[action]
    return td_error**2


@vmap
def double_q_learning_loss(q, target_q, action, action_select, reward, done, gamma):
    """Compute double q-learning loss through TD-learning (action selected by policy network)."""
    td_target = reward + gamma * target_q[action_select] * (1.0 - done)
    td_error = jax.lax.stop_gradient(td_target) - q[action]
    return td_error**2


@jit
def train_step(trainstate, target_params, batch, gamma=0.9):
    """Perform a single training step, i.e. compute loss and update model parameters."""

    def loss_fn(policy_params):
        """Compute avg loss for a batch of experiences."""
        state, action, reward, next_state, done = batch
        q = trainstate.apply_fn(policy_params, state)
        target_q = trainstate.apply_fn(target_params, next_state)
        action_select = trainstate.apply_fn(policy_params, next_state).argmax(-1)
        g = jnp.array([gamma] * state.shape[0])
        return jnp.mean(
            double_q_learning_loss(q, target_q, action, action_select, reward, done, g)
        )

    # compute loss and gradients, then apply gradients
    loss, grad = jax.value_and_grad(loss_fn)(trainstate.params)
    trainstate = trainstate.apply_gradients(grads=grad)
    return trainstate, loss
