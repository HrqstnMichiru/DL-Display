from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import json
import torch
from scipy.io.wavfile import write
from .env import AttrDict

from .model import Generator


h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    config_file = "D:\图片\临时\config_g.json"
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    generator = Generator(h).to(device)
    generator.load_state_dict(checkpoint_dict["generator"])
    return generator
