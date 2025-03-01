import argparse
from itertools import cycle

import diffrax
import heimburg_jackson.io as io
import jax.numpy as jnp
import matplotlib.pyplot as plt
from heimburg_jackson.conversions import Phi2u
from heimburg_jackson.models import iHJMicro
from jax.numpy.fft import fft, ifft
from matplotlib import colormaps
from matplotlib.lines import Line2D


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type",
        help="Specify the plot type",
        choices=["dispersion", "wave", "tip-speed", "symmetry", "animate", "summary"],
    )
    parser.add_argument("input", type=str, help="Path to the computation file")
    parser.add_argument(
        "-t",
        "--times",
        nargs="+",
        type=float,
        help="Times in the units of which the equation was solved"
        " for plotting the `wave` plot",
    )
    parser.add_argument(
        "--sweep-params",
        nargs="+",
        type=float,
        help="Sweep parameter values to use for plotting",
    )
    parser.add_argument(
        "--half-space", action="store_true", help="Plot only the left half space"
    )
    parser.add_argument(
        "-k",
        nargs=3,
        type=float,
        help="minimum, maximum and number of wave numbers for the dispersion plot.",
    )
    parser.add_argument(
        "-v",
        nargs=3,
        type=float,
        help="minimum, maximum and number of phase speeds for the dispersion plot.",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=False, help="Path to the output file"
    )
    parser.add_argument(
        "-b",
        "--baseline",
        type=str,
        required=False,
        help="Path to the baseline computation file",
    )
    parser.add_argument(
        "--xrange", nargs=2, type=float, help="Space range for plotting the wave"
    )

    args = parser.parse_args()
    return args


def _get_timestep_ids(args, ts) -> list:
    t_ids = []
    if args.times is not None:
        for t in args.times:
            if t == -1:
                step = (len(ts) - 1,)
            else:
                step = jnp.where(jnp.isclose(ts, t))[0]
                if step.shape[0] == 0:
                    print(
                        f"Time {t} is not recorded in file. Available times are {ts}"
                        " and -1 for last step"
                    )
                    exit(1)
            t_ids.append(step[0])
    return t_ids


def _get_sweep_param_ids(args, models_arrays, sweep_param: str | None) -> list:
    sp_ids = [0]
    if sweep_param is None:
        return sp_ids
    if args.sweep_params is None:
        return sp_ids

    models_sps = []
    for model, ys in models_arrays:
        models_sps.append(getattr(model, sweep_param))
    models_sps = jnp.array(models_sps)

    sp_ids = []
    for spv in args.sweep_params:
        id = jnp.where(jnp.isclose(models_sps, spv))[0]
        if id.shape[0] == 0:
            print(
                f"{sweep_param} {spv} is not recorded in the file."
                f" Available options are {models_sps}"
            )
            exit(1)
        sp_ids.append(id[0])
    return sp_ids


def _get_tip_speed(ts, ys, model):
    u = Phi2u(ys[:, 0, :], model.h2, model.K)[:, : model.nx // 2]  # time, phi, space
    x = model.x[jnp.argmax(u, axis=-1)]
    speeds = -jnp.diff(x) / jnp.diff(ts)
    return jnp.mean(speeds)


def plot_wave(
    axis, model, ys, t_ids, ts, sweep_param, label="", half_space=True, **plot_kw
):
    for t_id in t_ids:
        lab = label + f" {ts[t_id]} ms"
        if sweep_param is not None:
            spv = getattr(model, sweep_param)
            lab += f", {sweep_param} {round(spv, 2)}"
        left_slice = slice(model.nx // 2) if half_space else slice(model.nx)
        axis.plot(
            model.x[left_slice],
            Phi2u(ys[t_id, 0], model.h2, model.K)[left_slice],
            label=lab,
            **plot_kw,
        )
    axis.set_xlabel("space [m]")
    axis.set_ylabel(r"dimensionless density $U$")


def plot_dispersion(axis, kl: list, vl: list, model: iHJMicro, color="tab:blue"):
    k = jnp.linspace(kl[0], kl[1], int(kl[2]))
    phase_speed = jnp.linspace(vl[0], vl[1], int(vl[2]))
    k, phase_speed = jnp.meshgrid(k, phase_speed)
    axis.contour(
        k, phase_speed, model.dispersion(phase_speed * k, k), [0], colors=color
    )

    axis.set_xlabel("Dimensionless wave number")
    axis.set_ylabel("Dimensionless phase speed")

    g = jnp.sqrt(model.h1 / model.h2)
    axis.hlines([model.gamma, g], kl[0], kl[1], linestyle=":", colors=color)


def plot_tip_speed(axis, sweep_param, sweep_param_ids, models_arrays, ts):
    if sweep_param is None:
        print("File must contain multiple computations")
        exit(1)

    tip_speeds = []
    sps = []
    for sp_id in sweep_param_ids:
        model, ys = models_arrays[sp_id]
        sps.append(getattr(model, sweep_param))
        tip_speeds.append(_get_tip_speed(ts, ys, model))
    axis.plot(sps, tip_speeds)
    axis.set_xlabel(f"{sweep_param}")
    axis.set_ylabel("Speed of the wave's tip [m/s]")


def plot_symmetry(axis, ts, ys, t_ids, model, sweep_param, label="", **plot_kw):
    for t_id in t_ids:
        left_slice = slice(model.nx // 2)
        U = Phi2u(ys[t_id, 0], model.h2, model.K)
        U_x = ifft(1j * model.K * fft(U)).real

        lab = label + f" {ts[t_id]} ms"
        if sweep_param is not None:
            spv = getattr(model, sweep_param)
            lab += f", {sweep_param} {round(spv, 2)}"
        axis.plot(U[left_slice], U_x[left_slice], label=lab, **plot_kw)

    axis.set_xlabel(r"Dimensionless density $U$")
    axis.set_ylabel(r"$U_x$")


def animate(sol: diffrax.Solution):
    pass


if __name__ == "__main__":
    args = parse_args()

    models_arrays, ts, sweep_param = io.load_computations(args.input)
    baseline = None
    baseline_ts = None
    if args.baseline is not None:
        baseline_models, baseline_ts, _ = io.load_computations(args.baseline)
        if len(baseline_models) != 1:
            print("Baseline file must contain exactly one computation")
            exit(1)
        baseline = baseline_models[0]

    t_ids = _get_timestep_ids(args, ts)
    sweep_param_ids = _get_sweep_param_ids(args, models_arrays, sweep_param)

    fig, axes = plt.subplots(constrained_layout=True)
    cmap = cycle(colormaps["tab10"].colors)  # pyright: ignore
    baseline_color = "grey"

    if args.type == "wave":
        for sp_id in sweep_param_ids:
            model, ys = models_arrays[sp_id]
            plot_wave(
                axes, model, ys, t_ids, ts, sweep_param, half_space=args.half_space
            )

        if baseline is not None:
            model, ys = baseline
            plot_wave(
                axes,
                model,
                ys,
                t_ids,
                baseline_ts,
                None,
                r"$a_1=0$",
                args.half_space,
                linestyle="--",
                color=baseline_color,
            )
        axes.set_xlim(*args.xrange)
        axes.legend()

    elif args.type == "dispersion":
        legend_handles = []
        for sp_id in sweep_param_ids:
            model, _ = models_arrays[sp_id]
            c = next(cmap)
            plot_dispersion(axes, args.k, args.v, model, color=c)

            if sweep_param is not None:
                spv = getattr(model, sweep_param)
                legend_handles.append(
                    Line2D([0], [0], color=c, label=f"{sweep_param} {round(spv, 2)}")
                )

        if baseline is not None:
            model, _ = baseline
            plot_dispersion(axes, args.k, args.v, model, color=baseline_color)
            legend_handles.append(
                Line2D([0], [0], color=baseline_color, label=r"$a_1=0$")
            )
        axes.legend(handles=legend_handles)

    elif args.type == "tip-speed":
        sp_ids = (
            range(len(models_arrays)) if args.sweep_params is None else sweep_param_ids
        )
        plot_tip_speed(axes, sweep_param, sp_ids, models_arrays, ts)

        if baseline is not None:
            model, ys = baseline
            axes.axhline(
                _get_tip_speed(baseline_ts, ys, model).item(),
                color=baseline_color,
                linestyle="--",
                label="baseline",
            )

    elif args.type == "symmetry":
        for sp_id in sweep_param_ids:
            model, ys = models_arrays[sp_id]
            plot_symmetry(axes, ts, ys, t_ids, model, sweep_param)

        if baseline is not None:
            model, ys = baseline
            plot_symmetry(
                axes,
                baseline_ts,
                ys,
                t_ids,
                model,
                None,
                r"$a_1=0$",
                color=baseline_color,
                linestyle="--",
            )
        axes.legend()

    elif args.type == "animate":
        print("Not implemented yet")
        exit(1)

    elif args.type == "summary":
        plt.close("all")
        fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(12, 6))

        for sp_id in sweep_param_ids:
            model, ys = models_arrays[sp_id]
            c = next(cmap)
            plot_wave(
                axes[0, 0],
                model,
                ys,
                t_ids,
                ts,
                sweep_param,
                half_space=True,
                color=c,
            )
            axes[0, 0].legend(title=f"{sweep_param}")

            plot_dispersion(axes[0, 1], args.k, args.v, model, color=c)
            plot_symmetry(axes[1, 1], ts, ys, t_ids, model, sweep_param)
        plot_tip_speed(axes[1, 0], sweep_param, sweep_param_ids, models_arrays, ts)
        axes[0, 0].set_xlim(*args.xrange)

        if baseline is not None:
            model, ys = baseline
            plot_wave(
                axes[0, 0],
                model,
                ys,
                t_ids,
                baseline_ts,
                None,
                r"$a_1=0$",
                True,
                linestyle="--",
                color=baseline_color,
            )
            plot_dispersion(axes[0, 1], args.k, args.v, model, color=baseline_color)
            plot_symmetry(
                axes[1, 1],
                baseline_ts,
                ys,
                t_ids,
                model,
                None,
                r"$a_1=0$",
                color=baseline_color,
                linestyle="--",
            )
            axes[1, 0].axhline(
                _get_tip_speed(baseline_ts, ys, model).item(),
                color=baseline_color,
                linestyle="--",
                label="baseline",
            )

    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output)
