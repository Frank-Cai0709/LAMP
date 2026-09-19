import math
import torch
from torch import nn
import torch.nn.functional as F


def unreliability_map(logits, map_type):
    probability = torch.sigmoid(logits)
    if map_type == "entropy":
        eps = torch.finfo(probability.dtype).eps
        probability = probability.clamp(eps, 1 - eps)
        return -(probability * probability.log() +
                 (1 - probability) * (1 - probability).log()) / math.log(2)
    if map_type == "confidence":
        return 1 - torch.abs(2 * probability - 1)
    if map_type == "reverse_attention":
        return 1 - probability
    if map_type == "wave":
        return torch.sin(math.pi * probability)
    raise ValueError(f"Unsupported ERAB map: {map_type}")


class ERABModule(nn.Module):
    """Stage-specific edge reliability-aware refinement module."""

    def __init__(self, in_ch, guide_ch, map_type="wave", reduction=4, kernel=5,
                 use_local_branch=True):
        super().__init__()
        mid_ch = max(4, in_ch // reduction)
        self.map_type = map_type
        self.use_local_branch = use_local_branch
        self.low_proj = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.guide_proj = nn.Conv2d(guide_ch, 1, 1, bias=True)
        self.dw3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False)
        self.dwh = nn.Conv2d(mid_ch, mid_ch, (1, kernel), padding=(0, kernel // 2),
                             groups=mid_ch, bias=False)
        self.dwv = nn.Conv2d(mid_ch, mid_ch, (kernel, 1), padding=(kernel // 2, 0),
                             groups=mid_ch, bias=False)
        self.norm = nn.GroupNorm(1, mid_ch)
        self.edge_proj = nn.Conv2d(mid_ch, in_ch, 1, bias=False)
        self.ctx_proj = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.act = nn.GELU()
        self.refine_scale = nn.Parameter(torch.tensor(0.5))
        self.last_unreliability = None

    def forward(self, low_feat, guide_feat, return_unreliability=False):
        guide = F.interpolate(self.guide_proj(guide_feat), size=low_feat.shape[-2:],
                              mode="bilinear", align_corners=True)
        unreliability = unreliability_map(guide, self.map_type)
        self.last_unreliability = unreliability
        x = self.low_proj(low_feat) * unreliability
        edge = self.dwh(x) + self.dwv(x)
        if self.use_local_branch:
            edge = edge + self.dw3(x)
        edge = self.edge_proj(self.act(self.norm(edge)))
        context = self.ctx_proj(low_feat)
        attention = torch.sigmoid(edge - context) - 0.5
        output = low_feat + self.refine_scale * low_feat * attention
        return (output, unreliability) if return_unreliability else output


class IdentityERAB(nn.Module):
    def forward(self, low_feat, guide_feat, return_unreliability=False):
        unreliability = torch.ones((low_feat.shape[0], 1, *low_feat.shape[-2:]),
                                   device=low_feat.device, dtype=low_feat.dtype)
        return (low_feat, unreliability) if return_unreliability else low_feat
