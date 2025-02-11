import argparse
from pathlib import Path

import heimburg_jackson.io as io
import jax.numpy as jnp
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    TqdmProgressMeter,
    Tsit5,
    diffeqsolve,
)


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


def solve(config, output_file, sweep_param=None):
    model = io.create_model(config)

    # Initial condition
    u0 = (
        config["initial"]["A0"]
        * model.rho0
        * sech2(config["initial"]["B0"] * model.x / model.l)
    )

    sol = diffeqsolve(
        ODETerm(model),
        Tsit5(),
        model.T[0],
        model.T[-1],
        0.01,
        model.initial(u0),
        saveat=SaveAt(ts=model.T),
        stepsize_controller=PIDController(
            rtol=config["solver"]["rtol"], atol=config["solver"]["atol"]
        ),
        progress_meter=TqdmProgressMeter(20),
        max_steps=None,
    )
    print(sol.result)

    io.save_computation(output_file, model, sol, sweep_param)

    return sol, model


if __name__ == "__main__":
    args = parse_args()

    if Path(args.output).exists():
        raise ValueError(f"Output file {args.output} already exists.")

    # load the model
    configs, sweep_param = io.load_configs(args.config)

    for config in configs:
        if sweep_param is not None:
            print(f"Solving with {sweep_param}: {config['model'][sweep_param]}")
        sol, ihj = solve(config, args.output, sweep_param)
