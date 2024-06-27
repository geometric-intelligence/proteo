import os

import pydantic
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yml')


class Config(pydantic.BaseModel):
    class Config:
        extra = 'allow'

    def __getitem__(self, item):
        return getattr(self, item)

    def update(self, new_config):
        for k, v in new_config.items():
            setattr(self, k, v)


def read_config_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as _:
        obj = yaml.safe_load(_.read())
        return Config.parse_obj(obj)
