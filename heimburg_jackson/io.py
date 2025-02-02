import copy
from dataclasses import fields
from pathlib import Path
from typing import Optional

import diffrax
import h5py
import jax.numpy as jnp
import numpy as np
import yaml
from jaxtyping import Array

from .conversions import T2t
from .models import iHJ, iHJMicro


def save_computation(
    path: Path | str,
    model: iHJ | iHJMicro,
    solution: diffrax.Solution,
    sweep_param: Optional[str] = None,
):
    path = Path(path)
    fileexists = path.exists()
    with h5py.File(path, "a") as f:
        f.attrs["type"] = type(model).__name__
        for param in fields(model):
            # exclude arrays and the sweep parameter from the overall attributes
            if not param.init or param.name == sweep_param:
                continue

            # check if the file has the same parameters with what we have computed
            if fileexists:
                if f.attrs[param.name] != getattr(model, param.name):
                    raise ValueError(
                        f"The file contains other parameters than is the"
                        f"current computation. Got {f.attrs[param.name]} in file and"
                        f"{getattr(model, param.name)} in the model for {param.name}"
                    )
            else:
                f.attrs[param.name] = getattr(model, param.name)

        if not fileexists:  # include times only once
            f["t"] = T2t(solution.ts, model.c0, model.l)
        if sweep_param is None:
            f["ys"] = solution.ys
            f.attrs["result"] = solution.result.__repr__()
        else:
            f.attrs["sweep_param"] = sweep_param

            value = getattr(model, sweep_param)
            f[f"ys_{value}"] = solution.ys
            f[f"ys_{value}"].attrs["result"] = solution.result.__repr__()
            f[f"ys_{value}"].attrs[sweep_param] = value


def load_configs(path: Path | str) -> tuple[list[dict], Optional[str]]:
    with open(path, "r") as f:
        config = yaml.full_load(f)

    sweep_params = _get_sweep_params(config["model"])
    if len(sweep_params) == 0:
        return [config], None
    if len(sweep_params) > 1:
        raise ValueError("Only one sweep parameter is allowed")
    sweep_param = sweep_params[0]

    configs = _partition_configs(config, sweep_param)

    return configs, sweep_param


def _partition_configs(config: dict, sweep_param: str) -> list[dict]:
    value_range = []
    for value in config["model"][sweep_param]:
        if type(value) is dict:
            value_range.extend(np.linspace(value["from"], value["to"], value["num"]))
        elif type(value) is list:
            value_range.extend(value)
        elif type(value) is float:
            value_range.append(value)

    configs = []
    for value in value_range:
        configs.append(copy.deepcopy(config))
        configs[-1]["model"][sweep_param] = value

    return configs


def _get_sweep_params(config: dict) -> list:
    sweep_params = []
    for key, value in config.items():
        if type(value) is list:
            sweep_params.append(key)

    return sweep_params


def _create_model_flat_config(flat_config: dict) -> iHJMicro:
    params = copy.deepcopy(flat_config)
    model_type = params.pop("type")
    model = eval(model_type)(**params)

    return model


def create_model(config: dict) -> iHJMicro:
    flat_config = {**config["model"]}
    flat_config.update({**config["numerical"]})
    flat_config.update({**config["characteristic_sizes"]})

    model = _create_model_flat_config(flat_config)

    return model


def load_computations(
    path: Path | str,
) -> tuple[list[tuple[iHJMicro, Array]], Array, Optional[str]]:
    """Loads a file containing a sweep of computations

    Args:
        path (Path | str): Path to the file

    Returns:
        tuple[list[tuple[iHJMicro, Array, Array]], Optional[str]]: the returned tuple
        contains the list of reinitialized models solved arrays and the times.
        Optionally the sweep parameter name is returned, otherwise `None`.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        base_config = dict(f.attrs.items())
        sweep_param = base_config.pop("sweep_param", None)

        ts = jnp.array(f["t"])
        models_arrays = []
        for dset in f.keys():
            if dset == "t":
                continue
            if sweep_param is None:
                config = base_config
            else:
                config = {**base_config, f"{sweep_param}": f[dset].attrs[sweep_param]}

            model = _create_model_flat_config(config)
            ys = jnp.array(f[dset])
            models_arrays.append((model, ys))

    return models_arrays, ts, sweep_param
