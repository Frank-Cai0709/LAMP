import torch
from torch import nn


class AFRBModule(nn.Module):
    def __init__(self, dec_ch, skip_ch, out_ch, reduction=4, use_refinement=False):
        super().__init__()
        mid_ch = max(4, out_ch // reduction)
        self.use_conf = use_refinement
        self.dec_proj = nn.Conv2d(dec_ch, out_ch, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_ch, mid_ch, 1, bias=False),
            nn.GELU(), nn.Conv2d(mid_ch, out_ch, 1, bias=True), nn.Sigmoid())
        if self.use_conf:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
                nn.Conv2d(out_ch, 1, 1, bias=True), nn.Sigmoid())
            self.conf_scale = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Conv2d(out_ch, out_ch, 1, bias=False)

    def forward(self, decoder, skip, unreliability=None):
        d, s = self.dec_proj(decoder), self.skip_proj(skip)
        refined_skip = self.channel_mlp(d + s) * s
        self.last_feature_discrepancy = (d - refined_skip).detach()
        output = d + refined_skip
        self.last_correction = None
        if self.use_conf and unreliability is not None:
            discrepancy = torch.tanh(s - d)
            correction = self.spatial_gate(discrepancy) * unreliability * discrepancy
            self.last_correction = correction.detach()
            output = output + self.conf_scale * correction
        return self.out_proj(output)


class AddFusion(nn.Module):
    def forward(self, decoder, skip, unreliability=None):
        self.last_feature_discrepancy = (decoder - skip).detach()
        return decoder + skip


class ConcatFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, decoder, skip, unreliability=None):
        return self.proj(torch.cat((decoder, skip), dim=1))


class AttentionGateFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.decoder_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.skip_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Sequential(nn.GELU(), nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, decoder, skip, unreliability=None):
        return decoder + self.gate(self.decoder_proj(decoder) + self.skip_proj(skip)) * skip


def build_fusion(kind, channels, use_refinement=False):
    if kind == "afrb":
        return AFRBModule(channels, channels, channels, use_refinement=use_refinement)
    if kind == "add":
        return AddFusion()
    if kind == "concat":
        return ConcatFusion(channels)
    if kind == "attention_gate":
        return AttentionGateFusion(channels)
    raise ValueError(f"Unsupported fusion: {kind}")
