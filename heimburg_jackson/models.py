import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq, ifft
from jaxtyping import Array, ArrayLike, Float

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

    def initial(self, u0: Float[Array, " nx"]) -> Float[Array, "2 nx"]:
        """Generates initial condition for the equation in a dimensionless form
        with zero initial speed.

        Args:
            u0 (ArrayLike): initial condition of the wave. Should be given with
            the dimensions g/m^2.

        Returns:
            ArrayLike: Initial condition in the form suitable for `self.__call__`.
        """
        Phi0 = u2Phi(u2U(u0, self.rho0), self.h2, self.K)
        Psi0 = jnp.zeros_like(Phi0)
        return jnp.stack([Phi0, Psi0])

    def phase_speed(self, k: ArrayLike, dimless: bool = True) -> ArrayLike:
        """Calculate the phase speed from the dispersion relation.

        Args:
            k (ArrayLike): wave numbers.
            dimless (bool, optional): whether the wave numbers are given
            in a dimensionless form. Defaults to True.

        Returns:
            ArrayLike: phase speed
        """

        def vp(k, c0, h1, h2):
            return jnp.sqrt((c0**2 + h1 * k) / (1 + h2 * k))

        if dimless:
            return vp(k, 1, self.h1, self.h2)

        h1 = self.h1 * (self.c0 * self.l) ** 2
        h2 = self.h2 * self.l**2
        return vp(k, self.c0, h1, h2)

    def __call__(self, t, S: Float[Array, "2 nx"], args) -> Float[Array, "2 nx"]:
        r"""Function used in solving the improved Heimburg-Jackson differential
        equation. Should be passed to a regular ODE solver.
        .. math::
            u_{tt} = [(1 + pu + qu^2)u_x]_x - h_1 u_{xxxx} + h_2 u_{xxtt}

        We take a new variable Phi
        .. math::
            \Phi = u - h_2 u_xx = F^-1(F(u)) - h_2 F^-1((ik)^2 F(u))
        F is the Fourier transform with respect to x. So the equation becomes
        .. math::
            \Phi_tt = (1 + pu + mu^2)u_xx + (p + 2qu)u^2_x - h_1 u_xxxx
        And the wave function u can be found as
        .. math::
            u = F^-1[F(\Phi)/1+h_2 k^2]

        For every point :math:`x_0` we need to solve a system
        .. math::
            \Phi_t(x_0) = \Psi(x_0)
            \Psi_t(x_0) = (1 + pu(x_0) + qu(x_0)^2)u_{xx}(x_0)
                        + (p + 2qu(x_0))u(x_0)^2_x - h_1 u_{xxxx}(x_0)

        The spatial derivatives are found using the property
        .. math::
            F(u^{(n)}) = (ik)^n F(u)

        Args:
            t: unused
            S (ArrayLike): S[0, :] contains :math:`\Phi` and S[1, :] :math:`\Psi`
            contains :math:`\Psi` for each point
            args: unused

        Returns:
            ArrayLike: The time derivative of S
        """
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


class iHJMicro(eqx.Module):
    # characteristic sizes
    c0: Float  # m/s
    rho0: Float  # g/m^2
    l: Float  # m

    # equation
    p: Float  # * c0**2 / rho0
    q: Float  # * c0**2 / rho0**2
    h1: Float  # * (c0 * l)**2
    h2: Float  # * l**2
    a1: Float  # * rho0 * c0**2 / l
    a2: Float  # * c0**2 / (l * rho0)
    gamma: Float  # * c0
    eta: Float  # * c0 / l

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

    def initial(self, u0: Float[Array, " nx"]) -> Float[Array, "4 nx"]:
        """Generates initial condition for the equation in a dimensionless form
        with zero initial speed.

        Args:
            u0 (ArrayLike): initial condition of the wave. Should be given with
            the dimensions g/m^2.

        Returns:
            ArrayLike: Initial condition in the form suitable for `self.__call__`.
        """
        Phi0 = u2Phi(u2U(u0, self.rho0), self.h2, self.K)
        Psi0 = jnp.zeros_like(Phi0)
        Gamma0 = jnp.zeros_like(Phi0)
        Lambda0 = jnp.zeros_like(Phi0)
        return jnp.stack([Phi0, Psi0, Gamma0, Lambda0])

    def phase_speed(self, k: ArrayLike, dimless: bool = True) -> ArrayLike:
        """Calculate the phase speed from the dispersion relation.

        Args:
            k (ArrayLike): wave numbers.
            dimless (bool, optional): whether the wave numbers are given
            in a dimensionless form. Defaults to True.

        Returns:
            ArrayLike: phase speed
        """

        def vp(k, c0, h1, h2):
            return jnp.sqrt((c0**2 + h1 * k) / (1 + h2 * k))

        if dimless:
            return vp(k, 1, self.h1, self.h2)

        h1 = self.h1 * (self.c0 * self.l) ** 2
        h2 = self.h2 * self.l**2
        return vp(k, self.c0, h1, h2)

    def __call__(self, t, S: Float[Array, "4 nx"], args) -> Float[Array, "4 nx"]:
        r"""Function used in solving the improved Heimburg-Jackson differential
        equation. Should be passed to a regular ODE solver.
        .. math::
            u_{tt} = [(1 + pu + qu^2)u_x]_x - h_1 u_{xxxx} + h_2 u_{xxtt} + a_1 \Gamma_x
            \Gamma_{tt} = \gamma \Gamma_{xx} - \eta \Gamma - a_2 u_x

        We take a new variable Phi
        .. math::
            \Phi = u - h_2 u_xx = F^-1(F(u)) - h_2 F^-1((ik)^2 F(u))
        F is the Fourier transform with respect to x. So the equation becomes
        .. math::
            \Phi_tt = (1 + pu + mu^2)u_xx + (p + 2qu)u^2_x - h_1 u_xxxx
        And the wave function u can be found as
        .. math::
            u = F^-1[F(\Phi)/1+h_2 k^2]

        For every point :math:`x_0` we need to solve a system
        .. math::
            \Phi_t(x_0) = \Psi(x_0)
            \Psi_t(x_0) = (1 + pu(x_0) + qu(x_0)^2)u_{xx}(x_0)
                        + (p + 2qu(x_0))u(x_0)^2_x - h_1 u_{xxxx}(x_0) +a_1\Gamma_x(x_0)
            \Gamma_t(x_0) = \Lambda(x_0)
            \Lambda_t(x_0) = \gamma \Gamma_{xx}(x_0) - \eta \Gamma(x_0) - a_2 u_x(x_0)

        The spatial derivatives are found using the property
        .. math::
            F(u^{(n)}) = (ik)^n F(u)

        Args:
            t: unused
            S (ArrayLike): S[0, :] contains :math:`\Phi`, S[1, :] contains :math:`\Psi`,
            S[2, :] contains :math:`\Gamma` and S[3, :] :math:`\Lambda`
            contains :math:`\Psi` for each point
            args: unused

        Returns:
            ArrayLike: The time derivative of S
        """
        Phi = S[0]
        Psi = S[1]
        Gamma = S[2]
        Lambda = S[3]

        u = Phi2u(Phi, self.h2, self.K)

        u_fft = fft(u)

        u_x = ifft(1j * self.K * u_fft).real
        u_xx = ifft(-(self.K**2) * u_fft).real
        u_xxxx = ifft(self.K**4 * u_fft).real

        Gamma_fft = fft(Gamma)

        Gamma_x = ifft(1j * self.K * Gamma_fft).real
        Gamma_xx = ifft(-(self.K**2) * Gamma_fft).real

        return jnp.stack(
            [
                Psi,
                (1 + self.p * u + self.q * u**2) * u_xx
                + (self.p + 2 * self.q * u) * u_x**2
                - self.h1 * u_xxxx
                + self.a1 * Gamma_x,
                Lambda,
                self.gamma * Gamma_xx - self.eta * Gamma - self.a2 * u_x,
            ]
        )
