from typing import NamedTuple, Sequence, Tuple, Optional

import numpy as np
from scipy.linalg import expm
from scipy.special import legendre
from scipy.signal import cont2discrete

import jax
import jax.numpy as jnp
import flax.linen as nn


class LMUCell(nn.Module):
    """Legendre Memory Unit (LMU) Cell"""

    input_size: int
    hidden_size: int
    memory_size: int
    theta: int
    dt: float = 1.0
    learn_a: bool = False
    learn_b: bool = False

    def setup(self):
        A, B = self.calc_AB()
        self.A = self.param("A", lambda _: self.calc_AB()[0]) if self.learn_a else A
        self.B = self.param("B", lambda _: self.calc_AB()[1]) if self.learn_b else B

        iz, hz, mz = self.input_size, self.hidden_size, self.memory_size

        self.e_x = self.param("e_x", nn.initializers.lecun_uniform(), (self.input_size, 1))
        self.e_h = self.param("e_h", nn.initializers.lecun_uniform(), (self.hidden_size, 1))
        self.e_m = self.param("e_m", nn.initializers.lecun_uniform(), (self.memory_size, 1))

        self.W_in = self.param("W_in", nn.initializers.xavier_normal(), (iz, hz))
        self.W_h = self.param("W_h", nn.initializers.xavier_normal(), (hz, hz))
        self.W_m = self.param("W_m", nn.initializers.xavier_normal(), (mz, hz))

    def calc_AB(self):
        Q = np.arange(self.memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing="ij")

        A = R * np.where(i < j, -1, (-1.0) ** (i - j + 1))  # (memory_size, memory_size)
        B = R * ((-1.0) ** Q)                               # (memory_size, 1)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))

        A, B, C, D, _ = cont2discrete((A, B, C, D), self.dt, method="zoh")

        return A, B.T
    
    def __call__(self, x: jnp.array, state: Tuple[jnp.ndarray, jnp.ndarray] = None):

        if state is None:
            x = x.reshape(1, -1) if x.ndim == 1 else x
            state = (jnp.zeros((x.shape[0], self.hidden_size)),
                     jnp.zeros((x.shape[0], self.memory_size)))
        h, m = state

        # compute input to the memory block
        u = x @ self.e_x + h @ self.e_h + m @ self.e_m  # (1,)

        # compute new memory state
        m = m @ self.A + u @ self.B  # (memory_size,)

        # compute new hidden state
        h = jnp.tanh(x @ self.W_in + h @ self.W_h + m @ self.W_m)  # (hidden_size,)

        return h, m


class LDNCell(nn.Module):
    """Legendre Delay Network Cell
    Thanks to: https://github.com/neuromorphs/ant21-legendre/blob/main/notebooks/Basic%20LMU%20implementation.ipynb
    """

    size_in: int
    q: int = 1
    theta: int = 4
    decay: float = 0.5
    dt: float = 1

    def _calculate_initial(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = jnp.zeros((self.q, self.q))
        B = jnp.zeros((self.q, 1))
        for i in range(self.q):
            B = B.at[i].set((-1.0) ** i * (2 * i + 1))
            for j in range(self.q):
                A = A.at[i, j].set(
                    (2 * i + 1) * (-1 if i < j else (-1.0) ** (i - j + 1))
                )
        return A / self.theta, B / self.theta

    def _calc_expm(self, A):
        return expm(A * self.dt)

    def setup(self):
        A, B = self._calculate_initial()
        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        result_shape = jax.ShapeDtypeStruct(A.shape, jnp.float32)
        self.Ad = jax.pure_callback(self._calc_expm, result_shape, A)
        self.Bd = jnp.dot(jnp.dot(jnp.linalg.inv(A), (self.Ad - jnp.eye(self.q))), B)
        
    
    def __call__(self, x: jnp.array, state: Optional[jnp.ndarray] = None):
        if state is None:
            state = jnp.zeros((self.q, self.size_in))
        
        # this code will be called every timestep
        new_state = (self.Ad @ state) + (self.Bd @ x[None, :])
        return new_state.T.flatten(), new_state


class LDNCellCompact(nn.Module):
    """Legendre Delay Network Cell
    Thanks to: https://github.com/neuromorphs/ant21-legendre/blob/main/notebooks/Basic%20LMU%20implementation.ipynb
    """

    size_in: int
    q: int = 1
    theta: int = 4
    decay: float = 0.5
    dt: float = 1

    def _calculate_initial(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = jnp.zeros((self.q, self.q))
        B = jnp.zeros((self.q, 1))
        for i in range(self.q):
            B = B.at[i].set((-1.0) ** i * (2 * i + 1))
            for j in range(self.q):
                A = A.at[i, j].set(
                    (2 * i + 1) * (-1 if i < j else (-1.0) ** (i - j + 1))
                )
        return A / self.theta, B / self.theta

    def _calc_expm(self, A):
        return expm(A * self.dt)

    @nn.compact
    def __call__(self, x: jnp.ndarray, state: Optional[jnp.ndarray] = None):
        A = self.param("A", lambda _: self._calculate_initial()[0])
        B = self.param("B", lambda _: self._calculate_initial()[1])

        if state is None:
            state = jnp.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        result_shape = jax.ShapeDtypeStruct(A.shape, jnp.float32)
        Ad = jax.pure_callback(self._calc_expm, result_shape, A)
        Bd = jnp.dot(jnp.dot(jnp.linalg.inv(A), (Ad - jnp.eye(self.q))), B)

        # this code will be called every timestep
        new_state = (Ad @ state) + (Bd @ x[None, :])
        return new_state.T.flatten(), new_state
