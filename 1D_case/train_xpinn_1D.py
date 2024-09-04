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
    model1 = PINN_FCN(1, 1)
    model1.to(device)
    model2 = PINN_FCN(1, 1)
    model2.to(device)
    criterion = torch.nn.MSELoss()
    opt1 = optim.Adam(model1.parameters(), lr=1e-3)
    opt2 = optim.Adam(model2.parameters(), lr=1e-3)

    x1 = (torch.rand(size=(1250, 1)) * torch.pi / 2).to(device).requires_grad_()
    sigma1 = cp_sigma(x1)
    source1 = 4 * torch.sin(2 * x1.detach())
    x2 = (torch.rand(size=(1250, 1)) * torch.pi / 2 + torch.pi / 2).to(device).requires_grad_()
    sigma2 = cp_sigma(x2)
    source2 = 4 * torch.sin(2 * x2.detach())
    zeros = torch.zeros_like(x1, device=x1.device)

    x_start = torch.tensor([[0.0]]).to(device)
    x_end = torch.tensor([[torch.pi]]).to(device)
    zeros_bc = torch.zeros_like(x_start, device=x_start.device)

    x_jump = torch.tensor([torch.pi / 2]).reshape(-1, 1).to(device).requires_grad_()
    zeros_jump = torch.zeros_like(x_jump, device=x_jump.device)

    for iter in range(5000):
        u_x1 = model1(x1)
        lap_u_x1 = laplace(u_x1, x1)
        loss_pde1 = criterion(lap_u_x1 * sigma1 + source1, zeros)

        u_x2 = model1(x2)
        lap_u_x2 = laplace(u_x2, x2)
        loss_pde2 = criterion(lap_u_x2 * sigma2 + source2, zeros)

        loss_pde = loss_pde1 + loss_pde2

        u_start = model1(x_start)
        u_end = model2(x_end)
        loss_bc = criterion(u_start, zeros_bc) + criterion(u_end, zeros_bc)

        u1_jump = model1(x_jump)
        u2_jump = model2(x_jump)
        grad_u1_jump = gradient(u1_jump, x_jump)
        grad_u2_jump = gradient(u2_jump, x_jump)
        loss_int = criterion(3 * grad_u1_jump - grad_u2_jump, zeros_jump)
        loss_con = criterion(u1_jump - u2_jump, zeros_jump)

        loss = torch.sqrt(loss_pde) + torch.sqrt(loss_bc) + torch.sqrt(loss_int) + torch.sqrt(loss_con)

        loss.backward()
        opt1.step()
        opt2.step()
        model1.zero_grad()
        model2.zero_grad()
        x1.grad = None
        x2.grad = None
        x_jump.grad = None

        if (iter + 1) % 500 == 0:
            print(f"iter: {iter} " +
                  f"loss: {loss} loss_pde: {loss_pde} loss_bc: {loss_bc} loss_int: {loss_int} loss_con:{loss_con}")

        if (iter + 1) % 5000 == 0:
            save_path = os.path.join('../saved_models/1D_case/', f'xPINN1_{iter + 1}.pt')
            torch.save(model1.state_dict(), save_path)
            save_path = os.path.join('../saved_models/1D_case/', f'xPINN2_{iter + 1}.pt')
            torch.save(model2.state_dict(), save_path)
