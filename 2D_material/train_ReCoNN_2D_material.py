import os
import torch.optim as optim
from torch import nn
from utils import *


class MLP_2D_InteriorMaterial(nn.Module):
    def __init__(self, imv: [], zsf: [], zsf_sig: []):
        super(MLP_2D_InteriorMaterial, self).__init__()
        self.imv = imv  # imv stands for interior material vertices
        self.num_imv = len(imv)
        self.zsf = zsf  # zsf stands for zero set function
        self.zsf_sig = zsf_sig  # zsf_phi stands for zero set function for singular part
        self.num_zsf = len(zsf)
        self.fcn_w = nn.Sequential(
            nn.Linear(2, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 2 * (self.num_zsf + 1)),
        )
        self.fcn_phis = nn.ModuleList([nn.Sequential(
            nn.Linear(2, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, self.num_zsf + 1),
        ) for _ in range(self.num_imv)])
        self.lmbd = nn.Parameter(0.5 * torch.ones(self.num_imv))

    def forward(self, x):
        # regular part (rp) / singular part (sp)
        w = x
        for layer in self.fcn_w:
            w = layer(w)
        rp = w[:, 0:1]
        sp = torch.zeros_like(x[:, 0:1], device=x.device)
        for i, x_i in enumerate(self.imv):
            r = self._r(x, i)
            yita = self._yita(r)
            x_ba = (x - x_i) / r

            phi = (x - x_i) / r
            for layer in self.fcn_phis[i]:
                phi = layer(phi)
            s = phi[:, 0:1]

            w_index = (i + 1) * (self.num_zsf + 1)
            rp = rp + yita * w[:, w_index:w_index + 1]
            for p, (varphi, varphi_sig) in enumerate(zip(self.zsf, self.zsf_sig)):
                rp = rp + (w[:, p + 1:p + 2] + w[:, w_index + p + 1:w_index + p + 2] * yita) * varphi(x)
                s = s + phi[:, p + 1:p + 2] * varphi_sig(x_ba)

            sp = sp + s * yita * (r ** self.lmbd[i])

        return rp + sp

    def phi_i_p(self, x, i: int, p: int):
        x_i = self.imv[i]
        fcn_phi = self.fcn_phis[i]

        r = self._r(x, i)
        phi = (x - x_i) / r
        for layer in fcn_phi:
            phi = layer(phi)

        return phi[:, p + 1:p + 2]

    def w_i_p(self, x, i: int, p: int):
        for layer in self.fcn_w:
            x = layer(x)
        w_index = (i + 1) * (self.num_zsf + 1)

        return x[:, w_index + p + 1:w_index + p + 2]

    def _r(self, x, i: int):
        x_i = self.imv[i]
        return torch.norm(x - x_i, p=2, dim=1, keepdim=True)

    @staticmethod
    def _yita(r):
        t = 2.5 * r - 1.25

        zeros = torch.zeros_like(t, device=t.device)
        ones = torch.ones_like(t, device=t.device)
        sublevel_mul = torch.where((t < 0) | (t > 1), zeros, ones)
        sublevel_add = torch.where(t > 1, -ones, zeros)

        t = t * sublevel_mul
        t = -6 * (t ** 5) + 15 * (t ** 4) - 10 * (t ** 3) + 1
        return t + sublevel_add




if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP_2D_InteriorMaterial([torch.tensor([[0.0, 0.0]], device=device)],
                                    [lambda x: x[:, 0:1], lambda x: x[:, 1:2]],
                                    [lambda x: x[:, 0:1], lambda x: x[:, 1:2]])
    model.to(device)
    # input = torch.tensor([[0.5, 0.5],
    #                       [-0.5, 0.5],
    #                       [0.5, -0.5]], device=device)
    # output = model(input)
    pass
