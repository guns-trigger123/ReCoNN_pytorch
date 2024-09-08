import os
import torch.optim as optim
from torch import nn
from utils import *


class MLP_2D_SMOOTH(nn.Module):
    def __init__(self, zsf: []):
        super(MLP_2D_SMOOTH, self).__init__()
        self.zsf = zsf  # zsf stands for zero set function
        self.num_zsf = len(zsf)
        self.fcn = nn.Sequential(
            nn.Linear(2, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, self.num_zsf + 1),
        )

    def forward(self, x):
        input = x
        for layer in self.fcn:
            x = layer(x)
        out = x[:, 0:1]
        for i, varphi in enumerate(self.zsf):
            out = out + x[:, i + 1:i + 2] * torch.abs(varphi(input))
        return out

    def ui(self, x, i: int):
        for layer in self.fcn:
            x = layer(x)
        out = x[:, i:i + 1]
        return out


def varphi(x):
    return x[:, 0:1] ** 2 + x[:, 1:2] ** 2 - 0.25


def sigma(x):
    sig = torch.ones_like(x[:, 0:1], device=x.device)
    sig_temp = 3 * torch.ones_like(x[:, 0:1], device=x.device)
    sig = torch.where(torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2) < 0.5, sig_temp, sig)
    return sig


def real(x):
    sig = sigma(x)
    term1 = (x[:, 0:1] ** 2 - 1)
    term2 = (x[:, 1:2] ** 2 - 1)
    term3 = (4 * (x[:, 0:1] ** 2) + 4 * (x[:, 1:2] ** 2) - 1)
    return term1 * term2 * term3 / sig


def source(x):
    return sigma(x) * laplace(real(x), x).detach()


def interface(theta):
    x1 = 0.5 * torch.cos(theta)
    x2 = 0.5 * torch.sin(theta)
    return torch.cat([x1, x2], dim=1)


def bc(NUM_BOUNDARY):
    return torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                      torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                      torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                      torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])


if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP_2D_SMOOTH([varphi])
    model.to(device)
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ExponentialLR(opt, gamma=1e-3 ** (1 / 25000))

    x = (torch.rand(size=(1000, 2)) * 2 - 1).to(device).requires_grad_()
    sig = sigma(x)
    src = source(x)
    zeros = torch.zeros_like(x[:, 0:1], device=x.device)

    x_int = interface(torch.rand(size=(1000, 1)) * 2 * torch.pi).to(device).requires_grad_()
    gp_x_int = gradient(varphi(x_int), x_int).detach()  # gp stands for grad_phi
    gnp_x_int = torch.sqrt(gp_x_int[:, 0:1] ** 2 + gp_x_int[:, 1:2] ** 2)  # gnp stands for grad_norm_phi
    nv_x_int = gp_x_int / gnp_x_int  # nv stands for normal vector, also miu(x) in the paper
    zeros_int = torch.zeros_like(x_int[:, 0:1], device=x_int.device)

    x_bc = bc(250).to(device)
    zeros_bc = torch.zeros_like(x_bc[:, 0:1], device=x_bc.device)

    for iter in range(50000):
        u_x = model(x)
        lap_u_x = laplace(u_x, x)
        loss_pde = criterion(lap_u_x * sig - src, zeros)

        u_bc = model(x_bc)
        loss_bc = criterion(u_bc, zeros_bc)

        u0_int = model.ui(x_int, 0)
        u1_int = model.ui(x_int, 1)
        grad_u0_int = gradient(u0_int, x_int)
        dd_u0_int = torch.sum(grad_u0_int * nv_x_int, dim=1, keepdim=True)
        loss_int = criterion(-2 * dd_u0_int + 4 * u1_int * gnp_x_int, zeros_int)

        loss = torch.sqrt(loss_pde) + 10 * torch.sqrt(loss_bc) + 3.1623 * torch.sqrt(loss_int)

        loss.backward()
        opt.step()
        if iter > 24999:
            sch.step()
        model.zero_grad()
        x.grad = None
        x_int.grad = None

        if (iter + 1) % 5000 == 0:
            print(f"iter: {iter} " +
                  f"loss: {loss} loss_pde: {loss_pde} loss_bc: {loss_bc} loss_int: {loss_int}")

        if (iter + 1) % 5000 == 0:
            save_path = os.path.join('../saved_models/2D_smooth/', f'ReCoNN_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)
