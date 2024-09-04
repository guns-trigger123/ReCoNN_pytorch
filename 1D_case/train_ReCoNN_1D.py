import os
import torch.optim as optim
from torch import nn
from utils import *


class MLP_1D(nn.Module):
    def __init__(self, start, end, jump_points: list, material: list):
        super(MLP_1D, self).__init__()
        self.jump_points = jump_points
        self.num_jump = len(jump_points)
        self.material = material
        self.material_intervel = [(jump_points[i], jump_points[i + 1]) for i in range(self.num_jump - 1)]
        self.material_intervel.insert(0, (start, jump_points[0]))
        self.material_intervel.append((jump_points[-1], end))
        self.fcn = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, self.num_jump + 1),
        )

    def forward(self, x):
        input = x
        for layer in self.fcn:
            x = layer(x)
        out = x[:, 0:1]
        for i, p in enumerate(self.jump_points):
            out = out + x[:, i + 1:i + 2] * torch.abs(input - p) / 2
        return out

    def ui(self, x, i: int):
        for layer in self.fcn:
            x = layer(x)
        out = x[:, i:i + 1]
        return out

    def sigma(self, x):
        sig = torch.zeros_like(x, device=x.device)
        for mat, itv in zip(self.material, self.material_intervel):
            mat_temp = torch.ones_like(x, device=x.device) * mat
            sig = torch.where((x >= itv[0]) & (x <= itv[1]), mat_temp, sig)
        return sig


if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP_1D(0, torch.pi, [torch.pi / 2], [3, 1])
    model.to(device)
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    x = (torch.rand(size=(2500, 1)) * torch.pi).to(device).requires_grad_()
    sigma = model.sigma(x)
    source = 4 * torch.sin(2 * x.detach())
    zeros = torch.zeros_like(x, device=x.device)

    x_jump = torch.tensor(model.jump_points).reshape(-1, 1).to(device).requires_grad_()
    zeros_jump = torch.zeros_like(x_jump, device=x_jump.device)

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

        u0_jump = model.ui(x_jump, 0)
        u1_jump = model.ui(x_jump, 1)
        grad_u0_jump = gradient(u0_jump, x_jump)
        loss_int = criterion(-2 * grad_u0_jump + 2 * u1_jump, zeros_jump)

        loss = torch.sqrt(loss_pde) + torch.sqrt(loss_bc) + torch.sqrt(loss_int)

        loss.backward()
        opt.step()
        model.zero_grad()
        x.grad = None
        x_jump.grad = None

        if (iter + 1) % 500 == 0:
            print(f"iter: {iter} " +
                  f"loss: {loss} loss_pde: {loss_pde} loss_bc: {loss_bc} loss_int: {loss_int}")

        if (iter + 1) % 5000 == 0:
            save_path = os.path.join('../saved_models/1D_case/', f'ReCoNN_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)
