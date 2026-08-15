import jax
import jax.numpy as jnp
import lqrax
import numpy as np

dt = 0.05
pm = lqrax.PointMassLQR(dt=dt, ndim=2)
# The state is (px, py, vx, vy)
# Let's see how it steps
x = jnp.array([0.0, 0.0, 1.0, 2.0])
u = jnp.array([0.5, 0.5])
x_next = pm.step(x, u)
print("x:", x)
print("u:", u)
print("x_next:", x_next)
