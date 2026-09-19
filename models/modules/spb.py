import torch
from torch import nn
from mamba_ssm import Mamba


class SPBLayer(nn.Module):
    """Stage-specific Subject Perception Module."""

    def __init__(self, input_dim, output_dim, groups=4, propagation="sequential",
                 propagate_finest=False, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.internal_dim = ((input_dim + groups - 1) // groups) * groups
        self.output_dim = output_dim
        self.groups = groups
        self.propagation = propagation
        self.propagate_finest = propagate_finest
        self.input_proj = (nn.Identity() if self.internal_dim == input_dim else
                           nn.Linear(input_dim, self.internal_dim))
        self.norm = nn.LayerNorm(self.internal_dim)
        self.mamba = Mamba(d_model=self.internal_dim // groups, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(self.internal_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.group_scale = nn.Parameter(torch.ones(groups))
        if groups == 4:
            self.cross_scale_23 = nn.Parameter(torch.tensor(0.1))
            self.cross_scale_34 = nn.Parameter(torch.tensor(0.1))
        else:
            self.cross_scale = nn.ParameterList([
                nn.Parameter(torch.tensor(0.1)) for _ in range(groups - 1)
            ])

    def _cross_scale(self, index):
        if self.groups == 4:
            if index == 1:
                return self.cross_scale_23
            if index == 2:
                return self.cross_scale_34
            return self.group_scale.new_tensor(0.1)
        return self.cross_scale[index - 1]

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        batch, channels = x.shape[:2]
        spatial = x.shape[2:]
        tokens = x.reshape(batch, channels, -1).transpose(-1, -2)
        tokens = self.input_proj(tokens)
        groups = torch.chunk(self.norm(tokens), self.groups, dim=2)
        outputs = []
        for index, group in enumerate(groups):
            if index == 0 and not self.propagate_finest:
                output = group
            else:
                source = group
                if self.propagation == "sequential" and outputs:
                    source = source + self._cross_scale(index) * outputs[-1]
                output = self.mamba(source) + self.skip_scale * group
                if index == self.groups - 1:
                    output = self.group_scale[index] * output
            outputs.append(output)
        output = self.proj(self.norm(torch.cat(outputs, dim=2)))
        return output.transpose(-1, -2).reshape(batch, self.output_dim, *spatial)


class ParallelPVMLayer(SPBLayer):
    def __init__(self, input_dim, output_dim, groups=4, **kwargs):
        super().__init__(input_dim, output_dim, groups=groups,
                         propagation="parallel", propagate_finest=True, **kwargs)


class StandardMambaLayer(nn.Module):
    """Single-scale Mamba comparator used by Tables 20--22."""
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(d_model=input_dim, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        batch, channels = x.shape[:2]
        spatial = x.shape[2:]
        tokens = x.reshape(batch, channels, -1).transpose(-1, -2)
        output = self.proj(self.mamba(self.norm(tokens)))
        return output.transpose(-1, -2).reshape(batch, -1, *spatial)
