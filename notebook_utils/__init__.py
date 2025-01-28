from functools import cache
from pathlib import Path

import torch
from dotenv import dotenv_values


@cache
def get_script_dir() -> Path:
    file_path = Path(__file__)
    directory = file_path.parent

    return directory


@cache
def get_config() -> dict[str, str]:
    script_dir = get_script_dir()
    config = dotenv_values(f"{script_dir}/.env")

    return config


def get_config_value(key: str, default_value: str | None = None) -> str | None:
    return get_config().get(key, default_value)


@cache
def get_hf_token_path() -> str:
    return get_config_value("HF_TOKEN_PATH", "")


@cache
def get_hf_token() -> str | None:
    hf_token_path = get_hf_token_path()
    hf_token = None

    with open(hf_token_path, "r") as file:
        hf_token = file.read()

    return hf_token


@cache
def get_torch_device() -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return device
