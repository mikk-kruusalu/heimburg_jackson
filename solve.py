import argparse

import heimburg_jackson.io as io
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Solve the improved Heimburg-Jackson equation
        with added microstructure"""
    )

    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to the output file"
    )

    args = parser.parse_args()
    return args


def sech2(x):
    return 1 / jnp.cosh(x) ** 2


if __name__ == "__main__":
    args = parse_args()

    # load the model
    config = io.load_config(args.config)
    ihj = io.create_model(config)

    # Initial condition
    u0 = (
        config["initial"]["A0"]
        * ihj.rho0
        * sech2(config["initial"]["B0"] * ihj.x / ihj.l)
    )

    sol = diffeqsolve(
        ODETerm(ihj),
        Tsit5(),
        ihj.T[0],
        ihj.T[-1],
        0.01,
        ihj.initial(u0),
        saveat=SaveAt(ts=ihj.T),
        stepsize_controller=PIDController(
            rtol=config["solver"]["rtol"], atol=config["solver"]["atol"]
        ),
        progress_meter=TqdmProgressMeter(20),
        max_steps=None,
    )
    print(sol.result)

    io.save_computation(args.output, ihj, sol)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    k = 2 * jnp.pi / x2X(jnp.linspace(0.01, 0.5, 1000), ihj.l)
    phase_speed = jnp.linspace(0, 300, 1000) / ihj.c0
    k, phase_speed = jnp.meshgrid(k, phase_speed)
    ax[0].contour(k, phase_speed, ihj.dispersion(phase_speed * k, k), [0])

    for i in range(0, ihj.T.shape[0]):
        ax[1].plot(ihj.x, Phi2u(sol.ys[i, 0], ihj.h2, ihj.K), label=f"{i}")  # type: ignore
    plt.legend()
    plt.show()
