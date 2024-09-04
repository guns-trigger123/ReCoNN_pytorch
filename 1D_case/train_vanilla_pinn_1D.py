import os
import torch.optim as optim
from torch import nn
from utils import *


class PINN_FCN(nn.Module):
    def __init__(self, data_dim, output_dim):
        super().__init__()
        num_nerons = 20
        self.fcn = nn.Sequential(
            nn.Linear(data_dim, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, output_dim),
        )

    def forward(self, x):
        for layer in self.fcn:
            x = layer(x)
        return x


def cp_sigma(x):
    jump_points = [torch.pi / 2]
    num_jump = len(jump_points)
    material = [3, 1]
    material_intervel = [(jump_points[i], jump_points[i + 1]) for i in range(num_jump - 1)]
    material_intervel.insert(0, (0, jump_points[0]))
    material_intervel.append((jump_points[-1], torch.pi))

    sig = torch.zeros_like(x, device=x.device)
    for mat, itv in zip(material, material_intervel):
        mat_temp = torch.ones_like(x, device=x.device) * mat
        sig = torch.where((x >= itv[0]) & (x <= itv[1]), mat_temp, sig)
    return sig


if __name__ == '__main__':
    device = torch.device('cuda')
    model = PINN_FCN(1, 1)
    model.to(device)
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    x = (torch.rand(size=(2500, 1)) * torch.pi).to(device).requires_grad_()
    sigma = cp_sigma(x)
    source = 4 * torch.sin(2 * x.detach())
    zeros = torch.zeros_like(x, device=x.device)

    x_start = torch.tensor([[0.0]]).to(device)
    x_end = torch.tensor([[torch.pi]]).to(device)
    zeros_bc = torch.zeros_like(x_start, device=x_start.device)

    for iter in range(5000):
        u_x = model(x)
        lap_u_x = laplace(u_x, x)
        loss_pde = criterion(lap_u_x * sigma + source, zeros)

        u_start = model(x_start)
        u_end = model(x_end)
        loss_bc = criterion(u_start, zeros_bc) + criterion(u_end, zeros_bc)

        loss = torch.sqrt(loss_pde) + torch.sqrt(loss_bc)

        loss.backward()
        opt.step()
        model.zero_grad()
        x.grad = None

        if (iter + 1) % 500 == 0:
            print(f"iter: {iter} " +
                  f"loss: {loss} loss_pde: {loss_pde} loss_bc: {loss_bc}")

        if (iter + 1) % 5000 == 0:
            save_path = os.path.join('../saved_models/1D_case/', f'PINN_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)
