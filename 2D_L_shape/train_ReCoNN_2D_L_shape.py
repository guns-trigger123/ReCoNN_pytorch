import os

import torch
import torch.optim as optim
from torch import nn
from utils import *


class MLP_2D_Lshape(nn.Module):
    def __init__(self, rc: []):
        super(MLP_2D_Lshape, self).__init__()
        self.rc = rc  # rc stands for re-entrant corner
        self.num_rc = len(rc)
        self.fcn_w = nn.Sequential(
            nn.Linear(2, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, self.num_rc + 1),
        )
        self.fcn_phis = nn.ModuleList([nn.Sequential(
            nn.Linear(2, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 1),
        ) for _ in range(self.num_rc)])
        self.lmbd = nn.Parameter(0.5 * torch.ones(self.num_rc))

    def forward(self, x):
        w = x
        for layer in self.fcn_w:
            w = layer(w)
        out = w[:, 0:1]
        for i, x_i in enumerate(self.rc):
            r = self._r(x, i)
            yita = self._yita(r)
            phi = (x - x_i) / r
            for layer in self.fcn_phis[i]:
                phi = layer(phi)
            out = out + yita * w[:, i + 1:i + 2] + (r ** self.lmbd[i]) * yita * phi

        return out

    def phi_i(self, x, i: int):
        x_i = self.rc[i]
        fcn_phi = self.fcn_phis[i]

        r = self._r(x, i)
        phi = (x - x_i) / r
        for layer in fcn_phi:
            phi = layer(phi)

        return phi

    def _r(self, x, i: int):
        x_i = self.rc[i]
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


def pde_weight(x, x_i):
    ones = torch.ones_like(x[:, 0:1], device=x.device)
    x_sqr = 40 * (torch.norm(x - x_i, p=2, dim=1, keepdim=True) ** 2).detach()
    return torch.where(x_sqr < 1, x_sqr, ones)


def s0(x):
    r = torch.norm(x, p=2, dim=1, keepdim=True)
    theta = torch.arctan(x[:, 1:2] / x[:, 0:1])
    zeros = torch.zeros_like(theta, device=theta.device)
    PIs = torch.pi * torch.ones_like(theta, device=theta.device)
    PIs_add = torch.where(x[:, 0:1] < 0, PIs, zeros)
    theta = theta + PIs_add

    return (r ** (2.0 / 3)) * torch.sin(2.0 / 3 * (theta + torch.pi / 2))


def real(x):
    return s0(x) * (x[:, 0:1] ** 2 - 1) * (x[:, 1:2] ** 2 - 1)


def source(x):
    return laplace(real(x), x).detach()


def domain(NUM_DOMAIN):
    x1 = torch.rand(size=(NUM_DOMAIN, 2))
    x2 = torch.rand(size=(NUM_DOMAIN, 2)) + torch.tensor([-1.0, 0.0])
    x3 = torch.rand(size=(NUM_DOMAIN, 2)) + torch.tensor([0.0, -1.0])
    return torch.cat([x1, x2, x3])


def bc(NUM_BOUNDARY):
    Vm = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [1.0, -1.0]])
    Hm = torch.tensor([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]])
    bcs = []
    for i in range(4):
        bcs.append(torch.cat([torch.zeros([NUM_BOUNDARY, 1]), torch.rand([NUM_BOUNDARY, 1])], dim=1) + Vm[i])
        bcs.append(torch.cat([torch.rand([NUM_BOUNDARY, 1]), torch.zeros([NUM_BOUNDARY, 1])], dim=1) + Hm[i])
    return torch.cat(bcs)


if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP_2D_Lshape([torch.tensor([[0.0, 0.0]], device=device)])
    model.to(device)
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ExponentialLR(opt, gamma=1e-3 ** (1 / 25000))

    x = domain(334).to(device).requires_grad_()
    src = source(x)
    pde_w = pde_weight(x, model.rc[0])
    zeros = torch.zeros_like(x[:, 0:1], device=x.device)

    x_bc = bc(125).to(device)
    zeros_bc = torch.zeros_like(x_bc[:, 0:1], device=x_bc.device)

    x_phi_start = torch.tensor([[-1.0, 0.0]], device=device)
    x_phi_end = torch.tensor([[0.0, -1.0]], device=device)
    zeros_phi = torch.zeros_like(x_phi_start[:, 0:1], device=x_phi_start.device)

    for iter in range(50000):
        u_x = model(x)
        lap_u_x = laplace(u_x, x)
        loss_pde = torch.mean(pde_w * ((lap_u_x - src) ** 2))

        u_bc = model(x_bc)
        loss_bc = criterion(u_bc, zeros_bc)

        phi_start = model.phi_i(x_phi_start, 0)
        phi_end = model.phi_i(x_phi_end, 0)
        loss_phi = criterion(phi_start, zeros_phi) + criterion(phi_end, zeros_phi)

        loss = torch.sqrt(loss_pde) + torch.sqrt(loss_bc) * 10 + torch.sqrt(loss_phi)

        loss.backward()
        opt.step()
        if iter > 24999:
            sch.step()
        model.zero_grad()
        x.grad = None

        if (iter + 1) % 5000 == 0:
            print(f"iter: {iter} " +
                  f"loss: {loss} loss_pde: {loss_pde} loss_bc: {loss_bc} loss_phi: {loss_phi}")

        if (iter + 1) % 5000 == 0:
            save_path = os.path.join('../saved_models/2D_L_shape/', f'ReCoNN_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)

    # input = torch.tensor([[0.5, 0.5],
    #                       [-0.5, 0.5],
    #                       [0.5, -0.5]], device=device)
    # output = real(input)

    # x = (torch.rand(size=(10, 2)) * 2 - 1).to(device).requires_grad_()
    # x_i = torch.tensor([[0.0, 0.0]], device=device)
    # pde_w = pde_weight(x,x_i)

    # x, y = torch.linspace(0, 1, 128), torch.linspace(-1, 1, 256)
    # input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    # input_x, input_y = input_x.reshape(-1, 1), input_y.reshape(-1, 1)
    # input1 = torch.cat([input_x, input_y], dim=1)
    # x, y = torch.linspace(-1, 0, 128), torch.linspace(0, 1, 128)
    # input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    # input_x, input_y = input_x.reshape(-1, 1), input_y.reshape(-1, 1)
    # input2 = torch.cat([input_x, input_y], dim=1)
    # input = torch.cat([input1, input2])
    # output = real(input)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(6, 5))
    # plt.scatter(x[:, 0:1].cpu().detach(),
    #             x[:, 1:2].cpu().detach(),
    #             s=0.5,
    #             # c=output.cpu().detach(),
    #             cmap='rainbow'
    #             )
    # plt.xlim(-1.1, 1.1)
    # plt.ylim(-1.1, 1.1)
    # plt.colorbar()
    # plt.show()

    pass
