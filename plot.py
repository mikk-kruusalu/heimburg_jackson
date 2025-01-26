import argparse

import diffrax
import heimburg_jackson.io as io
import jax.numpy as jnp
import matplotlib.pyplot as plt
from heimburg_jackson.conversions import Phi2u


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type",
        help="Specify the plot type",
        choices=["dispersion", "animate", "wave", "summary"],
    )
    parser.add_argument("input", type=str, help="Path to the computation file")
    parser.add_argument(
        "-t",
        "--times",
        nargs="+",
        type=float,
        help="Times in the units of which the equation was solved"
        " for plotting the `wave` and `summary` plot",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=False, help="Path to the output file"
    )

    args = parser.parse_args()
    return args


def animate(sol: diffrax.Solution):
    pass


if __name__ == "__main__":
    args = parse_args()

    models_arrays, ts, sweep_param = io.load_computations(args.input)

    fig, axes = plt.subplots(constrained_layout=True)
    if args.type == "wave":
        steps = []
        for t in args.times:
            step = jnp.where(jnp.isclose(ts, t))[0]
            if step.shape[0] == 0:
                print(f"Time {t} is not recorded in file. Available times are {ts}")
                exit(1)
            steps.append(step[0])

        for step in steps:
            model, ys = models_arrays[step]
            axes.plot(
                model.x, Phi2u(ys[step, 0], model.h2, model.K), label=f"{ts[step]} ms"
            )
        axes.legend()
        axes.set_xlabel("space [m]")
        axes.set_ylabel("dimensionless density")
    elif args.type == "dispersion":
        print("Not implemented yet")
        exit(1)
    elif args.type == "animate":
        print("Not implemented yet")
        exit(1)
    elif args.type == "summary":
        print("Not implemented yet")
        exit(1)

    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output)
