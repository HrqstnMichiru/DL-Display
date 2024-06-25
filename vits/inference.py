import torch
from . import commons
from .text import text_to_sequence
from . import utils
from .models import SynthesizerTrn
from .text.symbols import symbols

def get_hps(filepath):
    hps = utils.get_hparams_from_file(filepath)
    return hps

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    print(text_norm)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def load_vits(hps):
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint("D:/图片/临时/G_12000.pth", net_g, None)
    return net_g
