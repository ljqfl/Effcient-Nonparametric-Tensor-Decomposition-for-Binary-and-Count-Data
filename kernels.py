import torch
import math
from typing import Optional
from torch import nn

def pairwise_distance(Xa, Xb):
    return torch.sum(Xa**2, dim=1, keepdim=True) + torch.sum(Xb**2, dim=1) - 2 * torch.mm(Xa, Xb.t())

def pairwise_inner(Xa, Xb):
    Xa_exp = Xa.unsqueeze(1)
    Xb_exp = Xb.unsqueeze(0)
    return (Xa_exp * Xb_exp).sum(-1)

def median(tensor):
    flat = tensor.view(-1)
    sorted_flat = torch.sort(flat)
    length = sorted_flat.size(0)
    mid = length // 2
    return (sorted_flat[mid] + sorted_flat[mid - 1]) / 2 if length % 2 == 0 else sorted_flat[mid]

class Kernel(nn.Module):
    pass

class RBF(Kernel):
    def __init__(self, band_width: Optional[float] = None):
        super().__init__()
        self.band_width = nn.Parameter(torch.tensor(band_width), requires_grad=False) if band_width is not None else None

    def forward(self, x, y=None):
        pdist = pairwise_distance(x, y if y is not None else x)
        sigma = median(pdist.detach()) / (2 * math.log(x.size(0) + 1)) if self.band_width is None else self.band_width ** 2
        return torch.exp(-0.5 * pdist / sigma)

class ARD(Kernel):
    def __init__(self, feature_len: int):
        super().__init__()
        self.log_band_width = nn.Parameter(torch.empty(feature_len))
        nn.init.normal_(self.log_band_width)

    def forward(self, x, y=None):
        bw = torch.exp(self.log_band_width)
        pdist = pairwise_distance_weight(x, y if y is not None else x, bw)
        return torch.exp(-0.5 * pdist)

class Mix(Kernel):
    def __init__(self, band_width: Optional[float] = None, r=0.1, c=0.0):
        super().__init__()
        self.band_width = nn.Parameter(torch.tensor(band_width), requires_grad=False) if band_width is not None else None
        self.r = r
        self.c = c

    def forward(self, x, y=None):
        pdist = pairwise_distance(x, y if y is not None else x)
        inner = pairwise_inner(x, y if y is not None else x)
        sigma = median(pdist.detach()) / (2 * math.log(x.size(0) + 1)) if self.band_width is None else self.band_width ** 2
        return torch.exp(-0.5 * pdist / sigma) + self.r * (inner + self.c) ** 2

if __name__ == "__main__":
    rbf = RBF(band_width=0.1)
    x = torch.randn(128, 5)
    y = torch.randn(64, 5)
    print(rbf(x, y))
