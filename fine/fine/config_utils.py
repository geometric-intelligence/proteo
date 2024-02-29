import os

import pydantic
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yml')


class Config(pydantic.BaseModel):
    """Proteo config."""

    class Config:
        extra = 'allow'


def read_config_from_file(path: str) -> 'Config':
    """Read config from yaml file.

    Parameters
    ----------
    path : str
        Path to the yaml file. Eg config.yml.

    Returns
    -------
    Config
        The config object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as _:
        obj = yaml.safe_load(_.read())
        return Config.parse_obj(obj)
