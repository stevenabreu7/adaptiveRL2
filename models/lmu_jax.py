from typing import NamedTuple, Sequence, Tuple, Optional

from scipy.linalg import expm
from scipy.special import legendre

import jax
import jax.numpy as jnp
import flax.linen as nn


class LMUCell(nn.Module):
    """Legendre Memory Unit (LMU) Cell
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

    @nn.compact
    def __call__(self, x: jnp.ndarray, state: Optional[jnp.ndarray] = None):
        A = self.param("A", lambda _: self._calculate_initial()[0])
        B = self.param("B", lambda _: self._calculate_initial()[1])

        if state is None:
            state = jnp.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = expm(A * self.dt)
        Bd = jnp.dot(jnp.dot(jnp.linalg.inv(A), (Ad - jnp.eye(self.q))), B)

        # this code will be called every timestep
        new_state = (Ad @ state) + (Bd @ x[None, :])
        return new_state.T.flatten(), new_state
