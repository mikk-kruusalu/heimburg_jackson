import jax
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
from jax.numpy.fft import fft, fftfreq, ifft
from jaxtyping import Array, Float


def Phi2u(Phi, h2, k):
    return ifft(fft(Phi) / (1 + h2 * k**2)).real


def u2Phi(u, h2, k):
    return u + h2 * ifft(k**2 * fft(u)).real


def u2U(u, rho0):
    return u / rho0


def U2u(U, rho0):
    return U * rho0


def x2X(x, l):
    return x / l


def X2x(X, l):
    return X * l


def t2T(t, c0, l):
    return t * c0 / l


def T2t(T, c0, l):
    return T * l / c0


def dSdt(t, S: Float[Array, "2 nx"], p: tuple) -> Float[Array, "2 nx"]:
    k, c0, p, q, h1, h2 = p

    Phi = S[0]
    Psi = S[1]

    u = Phi2u(Phi, h2, k)

    u_fft = fft(u)

    u_x = ifft(1j * k * u_fft).real
    u_xx = ifft(-(k**2) * u_fft).real
    u_xxxx = ifft(k**4 * u_fft).real

    return jnp.stack(
        [
            Psi,
            (c0**2 + p * u + q * u**2) * u_xx + (p + 2 * q * u) * u_x**2 - h1 * u_xxxx,
        ]
    )


def sech2(x):
    return 1 / jnp.cosh(x) ** 2


def test_diff(u0, k, order, period):
    from scipy.fftpack import diff

    u0diff = diff(u0, order, period=period)
    u0d = ifft((1j * k) ** order * fft(u0)).real
    print(u0d)

    _, ax = plt.subplots(1, 2)
    ax[0].plot(x, u0, label="u0")
    ax[0].plot(x, u0d, label="u0 derivative")
    ax[0].plot(x, u0diff, label="u0 scipy diff")
    ax[0].legend()

    ax[1].plot(x, u0diff - u0d)

    plt.show()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    c0 = 171.4  # m/s
    rho0 = 4.107e-3  # g/m^2
    l = 1e-2  # m

    # kriitiline väärtus on -3.32
    p = -3.32  # * c0**2 / rho0
    q = 32.32  # * c0**2 / rho0**2
    h1 = 2.0 / (c0 * l) ** 2  # * (c0 * l)**2
    h2 = 2.0 / l**2  # * l**2

    nx = 2**13
    domainL = 20  # m
    x = jnp.linspace(-domainL, domainL, nx, dtype="float64")
    # k = fftfreq(nx, d=domainL / jnp.pi, dtype="float64") * nx
    K = fftfreq(nx, d=x2X(domainL / jnp.pi, l), dtype="float64") * nx
    T = jnp.linspace(0, t2T(15e-3, c0, l), 9)

    u0 = rho0 / 2 * sech2(0.2 * x / l)
    Phi0 = u2Phi(u2U(u0, rho0), h2, K)
    Psi0 = jnp.zeros_like(Phi0)

    # print(T)
    # plt.plot(x2X(x, l), u2U(u0, rho0))
    # plt.show()
    # test_diff(Phi0, K, 2, x2X(2 * domainL, l))

    sol = diffeqsolve(
        ODETerm(dSdt),
        Tsit5(),
        T[0],
        T[-1],
        0.01,
        jnp.stack([Phi0, Psi0]),
        args=(K, 1, p, q, h1, h2),
        saveat=SaveAt(ts=T),
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-8),
        progress_meter=TqdmProgressMeter(20),
        max_steps=None,
    )
    print(sol.result)
    print(sol.ys.shape)  # type: ignore
    for i in range(0, T.shape[0] + 1, 3):
        plt.plot(x, Phi2u(sol.ys[i, 0], h2, K), label=f"{i}")  # type: ignore
    plt.legend()
    plt.show()
