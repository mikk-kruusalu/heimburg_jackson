from jax.numpy.fft import fft, ifft


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
