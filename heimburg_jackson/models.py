import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq, ifft
from jaxtyping import Array, Float

from .conversions import Phi2u, t2T, u2Phi, u2U, x2X


jax.config.update("jax_enable_x64", True)  # needed for accurate derivatives with fft


class iHJ(eqx.Module):
    # characteristic sizes
    c0: Float  # m/s
    rho0: Float  # g/m^2
    l: Float  # m

    # equation
    p: Float  # * c0**2 / rho0
    q: Float  # * c0**2 / rho0**2
    h1: Float  # * (c0 * l)**2
    h2: Float  # * l**2

    # grid
    nx: Float
    nt: Float
    domainL: Float  # m
    end_time: Float  # s

    x: Float[Array, ""] = eqx.field(init=False)
    K: Float[Array, ""] = eqx.field(init=False)
    T: Float[Array, ""] = eqx.field(init=False)

    def __post_init__(self):
        self.x = jnp.linspace(-self.domainL, self.domainL, self.nx, dtype="float64")
        self.K = (
            fftfreq(self.nx, d=x2X(self.domainL / jnp.pi, self.l), dtype="float64")
            * self.nx
        )
        self.T = jnp.linspace(0, t2T(self.end_time, self.c0, self.l), self.nt)

    def initial(self, u0):
        """Generates initial condition for the equation in a dimensionless form"""
        Phi0 = u2Phi(u2U(u0, self.rho0), self.h2, self.K)
        Psi0 = jnp.zeros_like(Phi0)
        return jnp.stack([Phi0, Psi0])

    def phase_speed(self, k, dimless=True):
        def vp(k, c0, h1, h2):
            return jnp.sqrt((c0**2 + h1 * k) / (1 + h2 * k))

        if dimless:
            return vp(k, 1, self.h1, self.h2)

        h1 = self.h1 * (self.c0 * self.l) ** 2
        h2 = self.h2 * self.l**2
        return vp(k, self.c0, h1, h2)

    def __call__(self, t, S, args):
        Phi = S[0]
        Psi = S[1]

        u = Phi2u(Phi, self.h2, self.K)

        u_fft = fft(u)

        u_x = ifft(1j * self.K * u_fft).real
        u_xx = ifft(-(self.K**2) * u_fft).real
        u_xxxx = ifft(self.K**4 * u_fft).real

        return jnp.stack(
            [
                Psi,
                (1 + self.p * u + self.q * u**2) * u_xx
                + (self.p + 2 * self.q * u) * u_x**2
                - self.h1 * u_xxxx,
            ]
        )
