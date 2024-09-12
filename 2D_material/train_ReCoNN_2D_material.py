import os

import torch
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
        r = self._r(x, 0)
        x_ba = (x - self.imv[0]) / r
        yita = self._yita(r)
        w = x
        for layer in self.fcn_w:
            w = layer(w)
        phi = x_ba
        for layer in self.fcn_phis[0]:
            phi = layer(phi)
        rp = w[:, 0:1] + w[:, 3:4] * yita
        rp = rp + (w[:, 1:2] + w[:, 4:5] * yita) * torch.abs(self.zsf[0](x))
        rp = rp + (w[:, 2:3] + w[:, 5:6] * yita) * torch.abs(self.zsf[1](x))
        sp = phi[:, 0:1]
        sp = sp + phi[:, 1:2] * torch.abs(self.zsf_sig[0](x_ba))
        sp = sp + phi[:, 2:3] * torch.abs(self.zsf_sig[1](x_ba))
        sp = sp * yita * (r ** self.lmbd[0])

        # w = x
        # for layer in self.fcn_w:
        #     w = layer(w)
        # rp = w[:, 0:1]
        # sp = torch.zeros_like(x[:, 0:1], device=x.device)
        # for i, x_i in enumerate(self.imv):
        #     r = self._r(x, i)
        #     yita = self._yita(r)
        #     x_ba = (x - x_i) / r
        #
        #     phi = (x - x_i) / r
        #     for layer in self.fcn_phis[i]:
        #         phi = layer(phi)
        #     s = phi[:, 0:1]
        #
        #     w_index = (i + 1) * (self.num_zsf + 1)
        #     rp = rp + yita * w[:, w_index:w_index + 1]
        #     for p, (varphi, varphi_sig) in enumerate(zip(self.zsf, self.zsf_sig)):
        #         rp = rp + (w[:, p + 1:p + 2] + w[:, w_index + p + 1:w_index + p + 2] * yita) * torch.abs(varphi(x))
        #         s = s + phi[:, p + 1:p + 2] * torch.abs(varphi_sig(x_ba))
        #
        #     sp = sp + s * yita * (r ** self.lmbd[i])

        return rp + sp

    def phi_p(self, x, p: int):
        # x_i = self.imv[0]
        # fcn_phi = self.fcn_phis[0]
        # phi = (x - x_i)
        # for layer in fcn_phi:
        #     phi = layer(phi)

        # x_i = self.imv[0]
        fcn_phi = self.fcn_phis[0]
        # r = self._r(x, 0)
        # phi = (x - x_i) / r
        phi = x
        for layer in fcn_phi:
            phi = layer(phi)
        return phi[:, p:p + 1]

    def w_p(self, x, p: int):
        r = self._r(x, 0)
        yita = self._yita(r)
        w_index = 3
        for layer in self.fcn_w:
            x = layer(x)
        return x[:, p:p + 1] + x[:, w_index + p:w_index + p + 1] * yita

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


def varphi0(x):
    return x[:, 0:1]


def varphi1(x):
    return x[:, 1:2]


def sigma(x):
    sig_i = [torch.ones_like(x[:, 0:1], device=x.device) * (i + 1) for i in range(4)]
    sig = torch.where((x[:, 0:1] < 0) & (x[:, 1:2] > 0), sig_i[1], sig_i[0])
    sig = torch.where((x[:, 0:1] < 0) & (x[:, 1:2] < 0), sig_i[2], sig)
    sig = torch.where((x[:, 0:1] > 0) & (x[:, 1:2] < 0), sig_i[3], sig)
    return sig


def s0(x):
    r = torch.norm(x, p=2, dim=1, keepdim=True)
    theta = - torch.arctan(x[:, 0:1] / x[:, 1:2])
    zeros = torch.zeros_like(theta, device=theta.device)
    PIs = torch.pi * torch.ones_like(theta, device=theta.device)
    PIs_add = torch.where(x[:, 1:2] < 0, PIs, zeros)
    theta = theta + PIs_add + torch.pi / 2

    lmbd = 0.8599
    theta = lmbd * theta

    ones = torch.ones_like(x[:, 0:1], device=x.device)
    zeros = torch.zeros_like(x[:, 0:1], device=x.device)
    quad1 = torch.where((x[:, 0:1] > 0) & (x[:, 1:2] > 0), ones, zeros)
    quad2 = torch.where((x[:, 0:1] < 0) & (x[:, 1:2] > 0), ones, zeros)
    quad3 = torch.where((x[:, 0:1] < 0) & (x[:, 1:2] < 0), ones, zeros)
    quad4 = torch.where((x[:, 0:1] > 0) & (x[:, 1:2] < 0), ones, zeros)

    s1 = (r ** lmbd) * (3.584 * torch.sin(theta) - 2.003 * torch.cos(theta)) * quad1
    s2 = (r ** lmbd) * (3.285 * torch.sin(theta) - 0.6678 * torch.cos(theta)) * quad2
    s3 = (r ** lmbd) * (2.474 * torch.sin(theta) - 1.0495 * torch.cos(theta)) * quad3
    s4 = (r ** lmbd) * (2.115 * torch.sin(theta) - 0.5861 * torch.cos(theta)) * quad4

    return s1 + s2 + s3 + s4


def real(x):
    return torch.cos(0.5 * torch.pi * x[:, 0:1]) * torch.cos(0.5 * torch.pi * x[:, 1:2]) * s0(x)


def pde_weight(x, x_i):
    ones = torch.ones_like(x[:, 0:1], device=x.device)
    x_sqr = 40 * (torch.norm(x - x_i, p=2, dim=1, keepdim=True) ** 2).detach()
    return torch.where(x_sqr < 1, x_sqr, ones)


def int_weight(x, x_i):
    ones = torch.ones_like(x[:, 0:1], device=x.device)
    x_abs = 40 * torch.norm(x - x_i, p=2, dim=1, keepdim=True).detach()
    return torch.where(x_abs < 1, x_abs, ones)


def source(x):
    return sigma(x) * laplace(real(x), x).detach()


def domain(NUM_DOMAIN: int):
    x1 = torch.rand(size=(NUM_DOMAIN, 2))
    x2 = torch.rand(size=(NUM_DOMAIN, 2)) + torch.tensor([-1.0, 0.0])
    x3 = torch.rand(size=(NUM_DOMAIN, 2)) + torch.tensor([0.0, -1.0])
    x4 = torch.rand(size=(NUM_DOMAIN, 2)) + torch.tensor([-1.0, -1.0])
    return torch.cat([x1, x2, x3, x4])


def bc(NUM_BOUNDARY: int):
    return torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                      torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                      torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                      torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])


def interface(NUM_INTERFACE: int, i: int):
    if i == 0:
        x_int = torch.cat([torch.zeros(NUM_INTERFACE, 1), torch.rand(NUM_INTERFACE, 1)], 1)
    elif i == 1:
        x_int = torch.cat([torch.rand(NUM_INTERFACE, 1), torch.zeros(NUM_INTERFACE, 1)], 1) + torch.tensor([-1.0, 0.0])
    elif i == 2:
        x_int = torch.cat([torch.zeros(NUM_INTERFACE, 1), torch.rand(NUM_INTERFACE, 1)], 1) + torch.tensor([0.0, -1.0])
    elif i == 3:
        x_int = torch.cat([torch.rand(NUM_INTERFACE, 1), torch.zeros(NUM_INTERFACE, 1)], 1)
    return x_int


if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP_2D_InteriorMaterial([torch.tensor([[0.0, 0.0]], device=device)],
                                    [varphi0, varphi1],
                                    [varphi0, varphi1])
    model.to(device)
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ExponentialLR(opt, gamma=1e-3 ** (1 / 25000))

    x = domain(250).to(device).requires_grad_()
    sig = sigma(x)
    src = source(x)
    pde_w = pde_weight(x, model.imv[0])

    x_bc = bc(250).to(device)
    zeros_bc = torch.zeros_like(x_bc[:, 0:1], device=x_bc.device)

    x_int0 = interface(250, 0).to(device).requires_grad_()
    # absv_int0 = torch.abs(varphi1(x_int0)).detach()
    gv_x_int0 = gradient(varphi0(x_int0), x_int0).detach()
    gnv_x_int0 = torch.norm(gv_x_int0, p=2, dim=1, keepdim=True)
    nv_x_int0 = gv_x_int0 / gnv_x_int0  # nv stands for normal vector
    int_w0 = int_weight(x_int0, model.imv[0])
    x_int1 = interface(250, 1).to(device).requires_grad_()
    # absv_int1 = torch.abs(varphi0(x_int1)).detach()
    gv_x_int1 = gradient(varphi1(x_int1), x_int1).detach()
    gnv_x_int1 = torch.norm(gv_x_int1, p=2, dim=1, keepdim=True)
    nv_x_int1 = gv_x_int1 / gnv_x_int1
    int_w1 = int_weight(x_int1, model.imv[0])
    x_int2 = interface(250, 2).to(device).requires_grad_()
    # absv_int2 = torch.abs(varphi1(x_int2)).detach()
    gv_x_int2 = gradient(varphi0(x_int2), x_int2).detach()
    gnv_x_int2 = torch.norm(gv_x_int2, p=2, dim=1, keepdim=True)
    nv_x_int2 = gv_x_int2 / gnv_x_int2
    int_w2 = int_weight(x_int2, model.imv[0])
    x_int3 = interface(250, 3).to(device).requires_grad_()
    # absv_int3 = torch.abs(varphi0(x_int3)).detach()
    gv_x_int3 = gradient(varphi1(x_int3), x_int3).detach()
    gnv_x_int3 = torch.norm(gv_x_int3, p=2, dim=1, keepdim=True)
    nv_x_int3 = gv_x_int3 / gnv_x_int3
    int_w3 = int_weight(x_int3, model.imv[0])

    x_phi0 = torch.tensor([[0.0, 1.0]], device=device).requires_grad_()  # already normalized
    # absv_phi0 = torch.abs(varphi1(x_phi0)).detach()
    gv_x_phi0 = gradient(varphi0(x_phi0), x_phi0).detach()
    gnv_x_phi0 = torch.norm(gv_x_phi0, p=2, dim=1, keepdim=True)
    nv_x_phi0 = gv_x_phi0 / gnv_x_phi0
    x_phi1 = torch.tensor([[-1.0, 0.0]], device=device).requires_grad_()
    # absv_phi1 = torch.abs(varphi0(x_phi1)).detach()
    gv_x_phi1 = gradient(varphi1(x_phi1), x_phi1).detach()
    gnv_x_phi1 = torch.norm(gv_x_phi1, p=2, dim=1, keepdim=True)
    nv_x_phi1 = gv_x_phi1 / gnv_x_phi1
    x_phi2 = torch.tensor([[0.0, -1.0]], device=device).requires_grad_()
    # absv_phi2 = torch.abs(varphi1(x_phi2)).detach()
    gv_x_phi2 = gradient(varphi0(x_phi2), x_phi2).detach()
    gnv_x_phi2 = torch.norm(gv_x_phi2, p=2, dim=1, keepdim=True)
    nv_x_phi2 = gv_x_phi2 / gnv_x_phi2
    x_phi3 = torch.tensor([[1.0, 0.0]], device=device).requires_grad_()
    # absv_phi3 = torch.abs(varphi0(x_phi3)).detach()
    gv_x_phi3 = gradient(varphi1(x_phi3), x_phi3).detach()
    gnv_x_phi3 = torch.norm(gv_x_phi3, p=2, dim=1, keepdim=True)
    nv_x_phi3 = gv_x_phi3 / gnv_x_phi3
    zeros_phi = torch.zeros_like(x_phi0[:, 0:1], device=x_phi0.device)

    for iter in range(50000):
        u_x = model(x)
        lap_u_x = laplace(u_x, x)
        loss_pde = torch.mean(pde_w * ((lap_u_x * sig - src) ** 2))

        u_bc = model(x_bc)
        loss_bc = criterion(u_bc, zeros_bc)

        w0_int0 = model.w_p(x_int0, 0)
        w1_int0 = model.w_p(x_int0, 1)
        w2_int0 = model.w_p(x_int0, 2)
        # r_int0 = torch.norm(x_int0 - model.imv[0], p=2, dim=1, keepdim=True)
        # yita_int0 = model._yita(r_int0)
        # phi0_int0 = model.phi_p(x_int0, 0) * yita_int0 * (r_int0 ** model.lmbd[0])
        # phi1_int0 = model.phi_p(x_int0, 1) * yita_int0 * (r_int0 ** model.lmbd[0])
        # phi2_int0 = model.phi_p(x_int0, 2) * yita_int0 * (r_int0 ** model.lmbd[0])
        # w0_int0 = w0_int0 + phi0_int0
        # w1_int0 = w1_int0 + phi1_int0
        # w2_int0 = w2_int0 + phi2_int0
        absv_int0 = torch.abs(varphi1(x_int0))
        grad_term0 = gradient(w0_int0 + w2_int0 * absv_int0, x_int0)
        dd_term0 = torch.sum(grad_term0 * nv_x_int0, dim=1, keepdim=True)  # directional derivative term
        loss_int0 = torch.mean(int_w0 * ((-1 * dd_term0 + 3 * w1_int0 * gnv_x_int0) ** 2))

        w0_int1 = model.w_p(x_int1, 0)
        w1_int1 = model.w_p(x_int1, 1)
        w2_int1 = model.w_p(x_int1, 2)
        # r_int1 = torch.norm(x_int1 - model.imv[0], p=2, dim=1, keepdim=True)
        # yita_int1 = model._yita(r_int1)
        # phi0_int1 = model.phi_p(x_int1, 0) * yita_int1 * (r_int1 ** model.lmbd[0])
        # phi1_int1 = model.phi_p(x_int1, 1) * yita_int1 * (r_int1 ** model.lmbd[0])
        # phi2_int1 = model.phi_p(x_int1, 2) * yita_int1 * (r_int1 ** model.lmbd[0])
        # w0_int1 = w0_int1 + phi0_int1
        # w1_int1 = w1_int1 + phi1_int1
        # w2_int1 = w2_int1 + phi2_int1
        absv_int1 = torch.abs(varphi0(x_int1))
        grad_term1 = gradient(w0_int1 + w1_int1 * absv_int1, x_int1)
        dd_term1 = torch.sum(grad_term1 * nv_x_int1, dim=1, keepdim=True)
        loss_int1 = torch.mean(int_w1 * ((-1 * dd_term1 + 5 * w2_int1 * gnv_x_int1) ** 2))

        w0_int2 = model.w_p(x_int2, 0)
        w1_int2 = model.w_p(x_int2, 1)
        w2_int2 = model.w_p(x_int2, 2)
        # r_int2 = torch.norm(x_int2 - model.imv[0], p=2, dim=1, keepdim=True)
        # yita_int2 = model._yita(r_int2)
        # phi0_int2 = model.phi_p(x_int2, 0) * yita_int2 * (r_int2 ** model.lmbd[0])
        # phi1_int2 = model.phi_p(x_int2, 1) * yita_int2 * (r_int2 ** model.lmbd[0])
        # phi2_int2 = model.phi_p(x_int2, 2) * yita_int2 * (r_int2 ** model.lmbd[0])
        # w0_int2 = w0_int2 + phi0_int2
        # w1_int2 = w1_int2 + phi1_int2
        # w2_int2 = w2_int2 + phi2_int2
        absv_int2 = torch.abs(varphi1(x_int2))
        grad_term2 = gradient(w0_int2 + w2_int2 * absv_int2, x_int2)
        dd_term2 = torch.sum(grad_term2 * nv_x_int2, dim=1, keepdim=True)
        loss_int2 = torch.mean(int_w2 * ((1 * dd_term2 + 7 * w1_int2 * gnv_x_int2) ** 2))

        w0_int3 = model.w_p(x_int3, 0)
        w1_int3 = model.w_p(x_int3, 1)
        w2_int3 = model.w_p(x_int3, 2)
        # r_int3 = torch.norm(x_int3 - model.imv[0], p=2, dim=1, keepdim=True)
        # yita_int3 = model._yita(r_int3)
        # phi0_int3 = model.phi_p(x_int3, 0) * yita_int3 * (r_int3 ** model.lmbd[0])
        # phi1_int3 = model.phi_p(x_int3, 1) * yita_int3 * (r_int3 ** model.lmbd[0])
        # phi2_int3 = model.phi_p(x_int3, 2) * yita_int3 * (r_int3 ** model.lmbd[0])
        # w0_int3 = w0_int3 + phi0_int3
        # w1_int3 = w1_int3 + phi1_int3
        # w2_int3 = w2_int3 + phi2_int3
        absv_int3 = torch.abs(varphi0(x_int3))
        grad_term3 = gradient(w0_int3 + w1_int3 * absv_int3, x_int3)
        dd_term3 = torch.sum(grad_term3 * nv_x_int3, dim=1, keepdim=True)
        loss_int3 = torch.mean(int_w3 * ((-3 * dd_term3 + 5 * w2_int3 * gnv_x_int3) ** 2))
        loss_int = (loss_int0 + loss_int1 + loss_int2 + loss_int3) / 4

        x_phi0_ba = (x_phi0 - model.imv[0]) / torch.norm(x_phi0 - model.imv[0], p=2, dim=1, keepdim=True)
        phi0_phi0 = model.phi_p(x_phi0_ba, 0)
        phi1_phi0 = model.phi_p(x_phi0_ba, 1)
        phi2_phi0 = model.phi_p(x_phi0_ba, 2)
        absv_phi0 = torch.abs(varphi1(x_phi0_ba))
        grad_phi_term0 = gradient(phi0_phi0 + phi2_phi0 * absv_phi0, x_phi0_ba)
        dd_phi_term0 = torch.sum(grad_phi_term0 * nv_x_phi0, dim=1, keepdim=True)
        loss_phi0 = criterion(-1 * dd_phi_term0 + 3 * phi1_phi0 * gnv_x_phi0, zeros_phi)

        x_phi1_ba = (x_phi1 - model.imv[0]) / torch.norm(x_phi1 - model.imv[0], p=2, dim=1, keepdim=True)
        phi0_phi1 = model.phi_p(x_phi1_ba, 0)
        phi1_phi1 = model.phi_p(x_phi1_ba, 1)
        phi2_phi1 = model.phi_p(x_phi1_ba, 2)
        absv_phi1 = torch.abs(varphi0(x_phi1_ba))
        grad_phi_term1 = gradient(phi0_phi1 + phi1_phi1 * absv_phi1, x_phi1_ba)
        dd_phi_term1 = torch.sum(grad_phi_term1 * nv_x_phi1, dim=1, keepdim=True)
        loss_phi1 = criterion(-1 * dd_phi_term1 + 5 * phi2_phi1 * gnv_x_phi1, zeros_phi)

        x_phi2_ba = (x_phi2 - model.imv[0]) / torch.norm(x_phi2 - model.imv[0], p=2, dim=1, keepdim=True)
        phi0_phi2 = model.phi_p(x_phi2_ba, 0)
        phi1_phi2 = model.phi_p(x_phi2_ba, 1)
        phi2_phi2 = model.phi_p(x_phi2_ba, 2)
        absv_phi2 = torch.abs(varphi1(x_phi2_ba))
        grad_phi_term2 = gradient(phi0_phi2 + phi2_phi2 * absv_phi2, x_phi2_ba)
        dd_phi_term2 = torch.sum(grad_phi_term2 * nv_x_phi2, dim=1, keepdim=True)
        loss_phi2 = criterion(1 * dd_phi_term2 + 7 * phi1_phi2 * gnv_x_phi2, zeros_phi)

        x_phi3_ba = (x_phi3 - model.imv[0]) / torch.norm(x_phi3 - model.imv[0], p=2, dim=1, keepdim=True)
        phi0_phi3 = model.phi_p(x_phi3_ba, 0)
        phi1_phi3 = model.phi_p(x_phi3_ba, 1)
        phi2_phi3 = model.phi_p(x_phi3_ba, 2)
        absv_phi3 = torch.abs(varphi0(x_phi3_ba))
        grad_phi_term3 = gradient(phi0_phi3 + phi1_phi3 * absv_phi3, x_phi3_ba)
        dd_phi_term3 = torch.sum(grad_phi_term3 * nv_x_phi3, dim=1, keepdim=True)
        loss_phi3 = criterion(-3 * dd_phi_term3 + 5 * phi2_phi3 * gnv_x_phi3, zeros_phi)
        loss_phi = (loss_phi0 + loss_phi1 + loss_phi2 + loss_phi3) / 4

        loss = torch.sqrt(loss_pde) + 3.1623 * torch.sqrt(loss_int) + 10 * torch.sqrt(loss_bc) + torch.sqrt(loss_phi)

        loss.backward()
        opt.step()
        if iter > 24999:
            sch.step()
        model.zero_grad()
        x.grad = None
        x_int0.grad = None
        x_int1.grad = None
        x_int2.grad = None
        x_int3.grad = None
        x_phi0.grad = None
        x_phi1.grad = None
        x_phi2.grad = None
        x_phi3.grad = None

        if (iter + 1) % 100 == 0:
            print(f"iter: {iter} " +
                  f"loss: {loss} loss_pde: {loss_pde} loss_bc: {loss_bc} loss_int: {loss_int} loss_phi: {loss_phi}")

        if (iter + 1) % 500 == 0:
            save_path = os.path.join('../saved_models/2D_material/', f'ReCoNN_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)
