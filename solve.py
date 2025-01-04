import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    TqdmProgressMeter,
    Tsit5,
    diffeqsolve,
)
from heimburg_jackson.conversions import Phi2u, x2X
from heimburg_jackson.models import iHJMicro


ihj = iHJMicro(
    c0=171.4,
    rho0=4.107e-3,
    l=1e-2,
    p=0.01,
    q=0.005,
    h1=0.2,
    h2=0.1,
    a1=0.7,
    a2=0.7,
    gamma=jnp.sqrt(0.4),
    eta=1.0,
    nx=2**13,
    domainL=20,
    end_time=25e-3,
    nt=9,
)


def sech2(x):
    return 1 / jnp.cosh(x) ** 2


u0 = ihj.rho0 / 2 * sech2(0.2 * ihj.x / ihj.l)


sol = diffeqsolve(
    ODETerm(ihj),
    Tsit5(),
    ihj.T[0],
    ihj.T[-1],
    0.01,
    ihj.initial(u0),
    saveat=SaveAt(ts=ihj.T),
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8),
    progress_meter=TqdmProgressMeter(20),
    max_steps=None,
)
print(sol.result)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

k = 2 * jnp.pi / x2X(jnp.linspace(0.01, 0.5, 1000), ihj.l)
phase_speed = jnp.linspace(0, 300, 1000) / ihj.c0
k, phase_speed = jnp.meshgrid(k, phase_speed)
ax[0].contour(k, phase_speed, ihj.dispersion(phase_speed * k, k), [0])

for i in range(0, ihj.T.shape[0] + 1, 3):
    ax[1].plot(ihj.x, Phi2u(sol.ys[i, 0], ihj.h2, ihj.K), label=f"{i}")  # type: ignore
plt.legend()
plt.show()
