import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .modules.afrb import build_fusion
from .modules.decoder import convolution
from .modules.erab import ERABModule, IdentityERAB
from .modules.spb import ParallelPVMLayer, SPBLayer, StandardMambaLayer


class LAMP(nn.Module):
    """Configurable LAMP network; defaults reproduce the released checkpoint."""

    def __init__(self, config):
        super().__init__()
        model = config["model"] if "model" in config else config
        channels = model.get("channels", [8, 16, 24, 32, 48, 64])
        self.c_list = channels
        self.return_sigmoid = True
        spb = model["spb"]
        erab = model["erab"]
        afrb = model["afrb"]
        self.encoder1 = convolution(model.get("input_channels", 3), channels[0])
        self.encoder2 = convolution(channels[0], channels[1])
        self.encoder3 = convolution(channels[1], channels[2])

        def semantic(in_ch, out_ch, location):
            if spb["enabled"] and location in spb.get("locations", []):
                kind = spb.get("kind", "spb")
                layer = {"spb": SPBLayer, "pvm": ParallelPVMLayer,
                         "mamba": StandardMambaLayer}[kind]
                kwargs = {} if kind == "mamba" else {
                    "groups": spb.get("groups", 4),
                    "propagation": spb.get("propagation", "sequential"),
                    "propagate_finest": spb.get("propagate_finest", False)}
                if kind == "pvm":
                    kwargs = {"groups": spb.get("groups", 4)}
                return nn.Sequential(layer(in_ch, out_ch, **kwargs))
            return convolution(in_ch, out_ch)

        self.encoder4 = semantic(channels[2], channels[3], "encoder4")
        self.encoder5 = semantic(channels[3], channels[4], "encoder5")
        self.encoder6 = semantic(channels[4], channels[5], "encoder6")
        self.decoder1 = semantic(channels[5], channels[4], "decoder1")
        self.decoder2 = semantic(channels[4], channels[3], "decoder2")
        self.decoder3 = semantic(channels[3], channels[2], "decoder3")
        self.decoder4 = convolution(channels[2], channels[1])
        self.decoder5 = convolution(channels[1], channels[0])

        erab_stages = set(erab.get("stages", [1, 2, 3])) if erab["enabled"] else set()
        make_erab = lambda stage, low, guide: ERABModule(
            low, guide, map_type=erab.get("map_type", "wave"),
            kernel=erab.get("kernel_size", 5),
            use_local_branch=erab.get("use_local_branch", True)
        ) if stage in erab_stages else IdentityERAB()
        self.heda3 = make_erab(3, channels[2], channels[3])
        self.heda2 = make_erab(2, channels[1], channels[2])
        self.heda1 = make_erab(1, channels[0], channels[1])
        self.heda4 = make_erab(4, channels[3], channels[4])

        fusion = afrb.get("fusion", "afrb") if afrb["enabled"] else "add"
        afrb_stages = set(afrb.get("stages", [1, 2, 3])) if afrb["enabled"] else set()
        refinement = set(afrb.get("refinement_stages", [3]))
        make_fusion = lambda stage, ch: build_fusion(
            fusion if stage in afrb_stages else "add", ch,
            use_refinement=stage in refinement)
        self.c3f3 = make_fusion(3, channels[2])
        self.c3f2 = make_fusion(2, channels[1])
        self.c3f1 = make_fusion(1, channels[0])
        self.c3f4 = make_fusion(4, channels[3])

        self.ebn1, self.ebn2, self.ebn3 = (nn.GroupNorm(4, c) for c in channels[:3])
        self.ebn4, self.ebn5 = nn.GroupNorm(4, channels[3]), nn.GroupNorm(4, channels[4])
        self.dbn1, self.dbn2 = nn.GroupNorm(4, channels[4]), nn.GroupNorm(4, channels[3])
        self.dbn3, self.dbn4, self.dbn5 = (nn.GroupNorm(4, c) for c in channels[2::-1])
        self.final = nn.Conv2d(channels[0], model.get("num_classes", 1), 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(0, math.sqrt(2.0 / (module.kernel_size[0] * module.out_channels)))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    @staticmethod
    def _up(block, norm, feature):
        return F.gelu(F.interpolate(norm(block(feature)), scale_factor=2,
                                    mode="bilinear", align_corners=True))

    def forward(self, x, return_maps=False):
        t1 = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2))
        t2 = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2))
        t3 = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2))
        t4 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(t3)), 2))
        t5 = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(t4)), 2))
        out = F.gelu(self.encoder6(t5))
        out5 = F.gelu(self.dbn1(self.decoder1(out))) + t5
        t4_ref, u4 = self.heda4(t4, out5, return_unreliability=True)
        out4 = self.c3f4(self._up(self.decoder2, self.dbn2, out5), t4_ref, u4)
        t3_ref, u3 = self.heda3(t3, out4, return_unreliability=True)
        out3 = self.c3f3(self._up(self.decoder3, self.dbn3, out4), t3_ref, u3)
        t2_ref, u2 = self.heda2(t2, out3, return_unreliability=True)
        out2 = self.c3f2(self._up(self.decoder4, self.dbn4, out3), t2_ref, u2)
        t1_ref, u1 = self.heda1(t1, out2, return_unreliability=True)
        out1 = self.c3f1(self._up(self.decoder5, self.dbn5, out2), t1_ref, u1)
        prediction = torch.sigmoid(F.interpolate(self.final(out1), scale_factor=2,
                                                 mode="bilinear", align_corners=True))
        if return_maps:
            return prediction, {"stage1": u1, "stage2": u2, "stage3": u3,
                                "stage4": u4}
        return prediction


def build_model(config):
    return LAMP(config)
