import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

import loralib as lora

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class VQModelTorch(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 rank=8,    # rank for lora
                 lora_alpha=1.0,
                 lora_tune_decoder=False,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        if lora_tune_decoder:
            conv_layer = partial(lora.Conv2d, r=rank, lora_alpha=lora_alpha)
        else:
            conv_layer = nn.Conv2d

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(rank=rank, lora_alpha=lora_alpha, lora_tune=lora_tune_decoder, **ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = conv_layer(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec

