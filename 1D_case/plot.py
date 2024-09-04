import matplotlib.pyplot as plt
from utils import *
from train_ReCoNN_1D import MLP_1D
from train_vanilla_pinn_1D import PINN_FCN


def real_u(x):
    jump_points = [torch.pi / 2]
    num_jump = len(jump_points)
    material = [3, 1]
    material_intervel = [(jump_points[i], jump_points[i + 1]) for i in range(num_jump - 1)]
    material_intervel.insert(0, (0, jump_points[0]))
    material_intervel.append((jump_points[-1], torch.pi))

    real = torch.zeros_like(x, device=x.device)
    for mat, itv in zip(material, material_intervel):
        mat_temp = torch.sin(2 * x) / mat
        real = torch.where((x >= itv[0]) & (x <= itv[1]), mat_temp, real)
    return real


def plot_ReCoNN():
    model = MLP_1D(0, torch.pi, [torch.pi / 2], [3, 1])
    model.load_state_dict(torch.load(("../saved_models/1D_case/ReCoNN_5000.pt")))

    x = torch.linspace(0, torch.pi, 1000).reshape(-1, 1)
    out = model(x).detach()
    ref = real_u(x)
    plt.figure(figsize=(6, 5))
    plt.plot(x.reshape(-1),
             out.reshape(-1),
             label="NN",
             )
    plt.plot(x.reshape(-1),
             ref.reshape(-1),
             label="REAL",
             )
    plt.legend()
    plt.title("ReCoNN")
    plt.show()


def plot_PINN():
    model = PINN_FCN(1, 1)
    model.load_state_dict(torch.load(("../saved_models/1D_case/PINN_5000.pt")))

    x = torch.linspace(0, torch.pi, 1000).reshape(-1, 1)
    out = model(x).detach()
    ref = real_u(x)
    plt.figure(figsize=(6, 5))
    plt.plot(x.reshape(-1),
             out.reshape(-1),
             label="NN",
             )
    plt.plot(x.reshape(-1),
             ref.reshape(-1),
             label="REAL",
             )
    plt.legend()
    plt.title("vanilla PINN")
    plt.show()


def plot_xPINN():
    model1 = PINN_FCN(1, 1)
    model1.load_state_dict(torch.load(("../saved_models/1D_case/xPINN1_5000.pt")))
    model2 = PINN_FCN(1, 1)
    model2.load_state_dict(torch.load(("../saved_models/1D_case/xPINN2_5000.pt")))

    x1 = torch.linspace(0, torch.pi / 2, 500).reshape(-1, 1)
    x2 = x1 + torch.pi / 2
    out1 = model1(x1).detach()
    out2 = model2(x2).detach()

    x = torch.cat([x1, x2], dim=0)
    out = torch.cat([out1, out2], dim=0)
    ref = real_u(x)

    plt.figure(figsize=(6, 5))
    plt.plot(x.reshape(-1),
             out.reshape(-1),
             label="NN",
             )
    plt.plot(x.reshape(-1),
             ref.reshape(-1),
             label="REAL",
             )
    plt.legend()
    plt.title("xPINN")
    plt.show()


if __name__ == '__main__':
    plot_ReCoNN()
    plot_PINN()
    plot_xPINN()
