"""Input/output utility functions."""

from os import PathLike
from pathlib import Path
from typing import Sequence, Union

import yaml

__all__ = [
    "readlines",
    "write_yaml",
    "load_yaml",
    "load_merge_yaml",
]


# TEXT
# ------------------------------------------------------------------------------
def readlines(file_path: Union[str, Path], encoding: str = None) -> list[str]:
    """Read a file into a list of strings, one per line.

    Args:
        file_path (Union[str, Path]): The path to the file.
        encoding (str, optional): The file encoding. Defaults to None.

    Returns:
        list[str]: A list of strings, one per line in the file.
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read().splitlines()


# YAML
# ------------------------------------------------------------------------------
def write_yaml(
    file_path: Union[str, Path],
    data: dict,
    create_dirs: bool = False,
    sort_keys: bool = False,
) -> None:
    """Write data to a YAML file.

    Args:
        file_path (Union[str, Path]): Path to the YAML file.
        data (dict): The data to write.
        create_dirs (bool, optional): Create parent directories if they don't exist. Defaults to False.
        sort_keys (bool, optional): Sort keys alphabetically in the output. Defaults to False.
    """
    file_path = Path(file_path).with_suffix(".yaml")
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def load_yaml(file_path: Union[str, Path]) -> dict:
    """Load data from a YAML file.

    Args:
        file_path (Union[str, Path]): Path to the YAML file.

    Returns:
        dict: The loaded data.
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_merge_yaml(*file_paths: Union[str, Path]) -> dict:
    """Load and merge multiple YAML files.

    This function loads YAML files in the given order, merging their
    contents recursively. Later files override earlier ones.

    Args:
        *file_paths (Union[str, Path]): Paths to the YAML files to load.

    Returns:
        dict: The merged configuration.

    Example:
        ```python
        merged_config = load_merge_yaml("base.yaml", "overrides.yaml")
        ```
    """
    merged_data = {}
    for file_path in file_paths:
        data = load_yaml(file_path)
        merged_data = _merge_dicts(merged_data, data)
    return merged_data


def _merge_dicts(old_dict: dict, new_dict: dict) -> dict:
    """Recursively merge two dictionaries.

    This is a helper function for `load_merge_yaml`. It merges the
    contents of two dictionaries recursively, with values from the
    `new_dict` taking precedence over those in `old_dict`.

    Args:
        old_dict (dict): The base dictionary.
        new_dict (dict): The dictionary to merge into the base.

    Returns:
        dict: The merged dictionary.
    """
    for key, value in new_dict.items():
        if (
            isinstance(value, dict)
            and key in old_dict
            and isinstance(old_dict[key], dict)
        ):
            old_dict[key] = _merge_dicts(old_dict[key], value)
        else:
            old_dict[key] = value
    return old_dict


# ------------------------------------------------------------------------------
