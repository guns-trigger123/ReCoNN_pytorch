import matplotlib.pyplot as plt
import torch

from utils import *
from train_ReCoNN_2D_L_shape import MLP_2D_Lshape


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


def plot_ReCoNN():
    model = MLP_2D_Lshape([torch.tensor([[0.0, 0.0]])])
    model.load_state_dict(torch.load(("../saved_models/2D_L_shape/ReCoNN_50000.pt")))

    x, y = torch.linspace(0, 1, 128), torch.linspace(-1, 1, 256)
    input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    input_x, input_y = input_x.reshape(-1, 1), input_y.reshape(-1, 1)
    input1 = torch.cat([input_x, input_y], dim=1)
    x, y = torch.linspace(-1, 0, 128), torch.linspace(0, 1, 128)
    input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    input_x, input_y = input_x.reshape(-1, 1), input_y.reshape(-1, 1)
    input2 = torch.cat([input_x, input_y], dim=1)
    input = torch.cat([input1, input2])

    out = model(input).detach()
    plt.figure(figsize=(6, 5))
    plt.scatter(input[:, 0:1],
                input[:, 1:2],
                s=0.5,
                c=out,
                # c=real(input),
                # c=torch.abs(out - real(input)),
                cmap='rainbow'
                )
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.colorbar()
    plt.title("ReCoNN")
    # plt.title("error")
    # plt.title("real")
    plt.show()


def plot_ReCoNN_phi():
    model = MLP_2D_Lshape([torch.tensor([[0.0, 0.0]])])
    model.load_state_dict(torch.load(("../saved_models/2D_L_shape/ReCoNN_50000.pt")))

    def interface(theta):
        x1 = 0.5 * torch.cos(theta)
        x2 = 0.5 * torch.sin(theta)
        return torch.cat([x1, x2], dim=1)

    theta = torch.linspace(-torch.pi / 2, torch.pi, 1000).reshape(-1, 1)
    x_int = interface(theta)

    out = model.phi_i(x_int, 0).detach()

    plt.figure(figsize=(6, 5))
    plt.plot(theta,
             out,
             label="NN",
             )
    plt.plot(theta,
             torch.sin(2.0 / 3 * (theta + torch.pi / 2)),
             label="REAL",
             )
    plt.title("ReCoNN phi")
    plt.legend()
    plt.show()


def plot_ReCoNN_lambda():
    model = MLP_2D_Lshape([torch.tensor([[0.0, 0.0]])])
    lmbd = []
    iterations = []
    for i in range(10):
        iter = 5000 * (i + 1)
        model.load_state_dict(torch.load((f"../saved_models/2D_L_shape/ReCoNN_{iter}.pt")))
        iterations.append(torch.tensor(iter).reshape(-1))
        lmbd.append(model.lmbd.clone().detach())

    iterations = torch.cat(iterations)
    lmbd = torch.cat(lmbd)
    plt.figure(figsize=(6, 5))
    plt.plot(iterations,
             lmbd,
             label="NN",
             )
    plt.plot(iterations,
             2.0 / 3 * torch.ones_like(iterations, dtype=torch.float32),
             label="REAL",
             )
    plt.title("ReCoNN lambda")
    plt.xlabel("iterations")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_ReCoNN()
    # plot_ReCoNN_phi()
    plot_ReCoNN_lambda()
