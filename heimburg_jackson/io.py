import copy
from dataclasses import fields
from pathlib import Path
from typing import Optional

import diffrax
import h5py
import numpy as np
import yaml

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
        for param in fields(model):
            if not param.init or param.name == sweep_param:
                continue

            if fileexists:
                if f.attrs[param.name] != getattr(model, param.name):
                    raise ValueError(
                        f"The file contains other parameters than is the"
                        f"current computation. Got {f.attrs[param.name]} in file and"
                        f"{getattr(model, param.name)} in the model for {param.name}"
                    )
            else:
                f.attrs[param.name] = getattr(model, param.name)

        if not fileexists:
            f["t"] = solution.ts
        if sweep_param is None:
            f["ys"] = solution.ys
            f.attrs["result"] = solution.result.__repr__()
        else:
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
            value_range.extend(np.arange(value["from"], value["to"], value["step"]))
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


def create_model(config: dict):
    params = {**config["model"]}
    params.pop("name")
    params.update({**config["numerical"]})
    params.update({**config["characteristic_sizes"]})

    model = eval(config["model"]["name"])(**params)

    return model
