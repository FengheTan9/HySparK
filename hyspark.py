from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import models.encoder as encoder
from models.decoder import LightDecoder


class HySparK(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, densify_norm='ln', sbn=True,
    ):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        self.fmap_h, self.fmap_w, self.fmap_d = input_size // downsample_raito, input_size // downsample_raito, input_size // downsample_raito
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * self.fmap_d * (1 - mask_ratio))

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder

        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(
                self.hierarchy):  # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)

            # create densify norm
            densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)

            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()  # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[HySparK.__init__, densify {i + 1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv3d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                         bias=True)
                print(
                    f'[HySparK.__init__, densify {i + 1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)

            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2

        print(f'[HySparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')

    def mask(self, B: int, device, generator=None):
        h, w, d = self.fmap_h, self.fmap_w, self.fmap_d
        idx = torch.rand(B, h * w * d, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, h * w * d, dtype=torch.bool, device=device)\
            .scatter_(dim=1, index=idx, value=True).view(B, 1, h, w, d)

    def forward(self, inp_bchwd: torch.Tensor, active_b1fff=None, vis=False):
        # step1. Mask
        if active_b1fff is None:  # rand mask
            active_b1fff: torch.BoolTensor = self.mask(inp_bchwd.shape[0], inp_bchwd.device)  # (B, 1, f, f, f)
        encoder._cur_active = active_b1fff  # (B, 1, f, f)
        active_b1hwd = active_b1fff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito,
                                                                                                  3).repeat_interleave(
            self.downsample_raito, 4)  # (B, 1, H, W, D)
        masked_bchwd = inp_bchwd * active_b1hwd

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcfffs: List[torch.Tensor] = self.sparse_encoder(masked_bchwd, active_b1fff)
        fea_bcfffs.reverse()  # after reversion: from the smallest feature map to the largest

        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1fff  # (B, 1, f, f, f)
        to_dec = []
        for i, bcfff in enumerate(fea_bcfffs):  # from the smallest feature map to the largest
            if bcfff is not None:
                bcfff = self.densify_norms[i](bcfff)
                mask_tokens = self.mask_tokens[i].expand_as(bcfff)
                bcfff = torch.where(cur_active.expand_as(bcfff), bcfff,
                                    mask_tokens)  # fill in empty (non-active) positions with [mask] tokens
                bcfff: torch.Tensor = self.densify_projs[i](bcfff)
            to_dec.append(bcfff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2,
                                                                                                              dim=4)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)

        # step4. Decode and reconstruct
        rec_bchwd = self.dense_decoder(to_dec)
        inp, rec = self.patchify(inp_bchwd), self.patchify(
            rec_bchwd)  # inp and rec: (B, L = f*f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)

        non_active = active_b1fff.logical_not().int().view(active_b1fff.shape[0], -1)  # (B, 1, f, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (
                    non_active.sum() + 1e-8)  # loss only on masked (non-active) patches

        if vis:
            masked_bchwd = inp_bchwd * active_b1hwd
            rec_bchwd = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hwd, inp_bchwd, rec_bchwd)
            return inp_bchwd, masked_bchwd, rec_or_inp
        else:
            return recon_loss

    def patchify(self, bchwd):
        p = self.downsample_raito
        h, w, d = self.fmap_h, self.fmap_w, self.fmap_d
        B, C = bchwd.shape[:2]
        bchwd = bchwd.reshape(shape=(B, C, h, p, w, p, d, p))
        bchwd = torch.einsum('bchpwqds->bhwdpqsc', bchwd)
        bln = bchwd.reshape(shape=(B, h * w * d, C * p ** 3))  # (B, f*f, 3*downsample_raito**2)
        return bln

    def unpatchify(self, bln):
        p = self.downsample_raito
        h, w, d = self.fmap_h, self.fmap_w, self.fmap_d
        B, C = bln.shape[0], bln.shape[-1] // p ** 3
        bln = bln.reshape(shape=(B, h, w, d, p, p, p, C))
        bln = torch.einsum('bhwdpqsc->bchpwqds', bln)
        bchwd = bln.reshape(shape=(B, C, h * p, w * p, d * p))
        return bchwd

    def __repr__(self):
        return (
            f'\n'
            f'[HySparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[HySparK.structure]: {super(HySparK, self).__repr__().replace(HySparK.__name__, "")}'
        )

    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,

            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(HySparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state

    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(HySparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
