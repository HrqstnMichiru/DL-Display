import matplotlib.pyplot as plt
import numpy as np
import torch
from zmq import HICCUP_MSG
from .hparams import create_hparams
from .model import Tacotron2
from .layers import TacotronSTFT, STFT
from .audio_processing import griffin_lim
from .text import text_to_sequence
from .denoiser import Denoiser
import soundfile as sf
from .hifi_gan.load import load_checkpoint
from numpy import finfo


def load_tacotron2():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    model = Tacotron2(hparams)
    model.to(device)
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo("float16").min
    checkpoint_path = "D:/图片/临时/checkpoint_28400"
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model = model.eval()
    return model

def load_waveglow():
    waveglow_path = "D:/图片/临时/waveglow_256channels_universal_v5.pt"
    waveglow = torch.load(waveglow_path)["model"]
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow,denoiser

def load_hifigan():
    hifi_path = "D:/图片/临时/g_02518000"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    HiFi_GAN = load_checkpoint(hifi_path, device)
    HiFi_GAN.remove_weight_norm()
    return HiFi_GAN
