from dataclasses import fields
from pathlib import Path

import diffrax
import h5py
import yaml

from .models import iHJ, iHJMicro


def save_computation(
    path: Path | str, model: iHJ | iHJMicro, solution: diffrax.Solution
):
    with h5py.File(path, "w") as f:
        f["t"] = solution.ts
        f["ys"] = solution.ys

        for param in fields(model):
            if not param.init:
                continue
            f.attrs[param.name] = getattr(model, param.name)
        f.attrs["result"] = solution.result.__repr__()


def load_config(path: Path | str) -> dict:
    with open(path, "r") as f:
        config = yaml.full_load(f)

    return config


def create_model(config: dict):
    params = {**config["model"]}
    params.pop("name")
    params.update({**config["numerical"]})
    params.update({**config["characteristic_sizes"]})

    model = eval(config["model"]["name"])(**params)

    return model
