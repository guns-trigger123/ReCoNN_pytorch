import matplotlib.pyplot as plt
from utils import *
from train_ReCoNN_2D_material import MLP_2D_InteriorMaterial


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

    a1 = 3.58396766000856
    a2 = 3.28530926421398
    a3 = 2.47465193074208
    a4 = 2.11503551932097
    b1 = -2.00351054735044
    b2 = -0.667838639957490
    b3 = -1.04946191312213
    b4 = -0.586064428952415

    lmbd = 0.8599513039
    theta = lmbd * theta

    ones = torch.ones_like(x[:, 0:1], device=x.device)
    zeros = torch.zeros_like(x[:, 0:1], device=x.device)
    quad1 = torch.where((x[:, 0:1] > 0) & (x[:, 1:2] > 0), ones, zeros)
    quad2 = torch.where((x[:, 0:1] < 0) & (x[:, 1:2] > 0), ones, zeros)
    quad3 = torch.where((x[:, 0:1] < 0) & (x[:, 1:2] < 0), ones, zeros)
    quad4 = torch.where((x[:, 0:1] > 0) & (x[:, 1:2] < 0), ones, zeros)

    s1 = (r ** lmbd) * (a1 * torch.sin(theta) + b1 * torch.cos(theta)) * quad1
    s2 = (r ** lmbd) * (a2 * torch.sin(theta) + b2 * torch.cos(theta)) * quad2
    s3 = (r ** lmbd) * (a3 * torch.sin(theta) + b3 * torch.cos(theta)) * quad3
    s4 = (r ** lmbd) * (a4 * torch.sin(theta) + b4 * torch.cos(theta)) * quad4

    return s1 + s2 + s3 + s4


def real(x):
    return torch.cos(0.5 * torch.pi * x[:, 0:1]) * torch.cos(0.5 * torch.pi * x[:, 1:2]) * s0(x)


def plot_ReCoNN(iteration, experiment):
    model = MLP_2D_InteriorMaterial([torch.tensor([[0.0, 0.0]])],
                                    [varphi0, varphi1],
                                    [varphi0, varphi1])
    model.load_state_dict(torch.load(("../saved_models/2D_material/" + experiment + f"/ReCoNN_{iteration}.pt")))

    x, y = torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256)
    input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    input_x, input_y = input_x.reshape(-1, 1), input_y.reshape(-1, 1)
    input = torch.cat([input_x, input_y], dim=1)

    out = model(input).detach()

    input_withgrad = input.clone().requires_grad_()
    out_withgrad = model(input_withgrad)
    lap_out_withgrad = laplace(out_withgrad, input_withgrad).detach()
    lap_real = laplace(real(input_withgrad), input_withgrad).detach()

    plt.figure(figsize=(6, 5))
    plt.scatter(input[:, 0:1],
                input[:, 1:2],
                s=0.5,
                # c=out,
                # c=real(input),
                c=torch.abs(out - real(input)),
                # c=lap_out_withgrad,
                # c=lap_real,
                # c=torch.abs(lap_real-lap_out_withgrad),
                cmap='rainbow'
                )
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.colorbar()
    # plt.clim(0,0.04)
    plt.title("ReCoNN")
    # plt.title("error")
    # plt.title("real")
    plt.show()


def plot_ReCoNN_phi(iteration, experiment):
    model = MLP_2D_InteriorMaterial([torch.tensor([[0.0, 0.0]])],
                                    [varphi0, varphi1],
                                    [varphi0, varphi1])
    model.load_state_dict(torch.load(("../saved_models/2D_material/" + experiment + f"/ReCoNN_{iteration}.pt")))

    def interface(theta):
        x1 = 1 * torch.cos(theta)
        x2 = 1 * torch.sin(theta)
        return torch.cat([x1, x2], dim=1)

    theta = torch.linspace(0, 2 * torch.pi, 1000).reshape(-1, 1)
    x = interface(theta)

    phi0 = model.phi_p(x, 0)
    phi1 = model.phi_p(x, 1)
    phi2 = model.phi_p(x, 2)
    out = model.phi_p(x, 0) + model.phi_p(x, 1) * torch.abs(varphi0(x)) + model.phi_p(x, 2) * torch.abs(varphi1(x))
    out = out.detach()

    ones = torch.ones_like(x[:, 0:1], device=x.device)
    zeros = torch.zeros_like(x[:, 0:1], device=x.device)
    quad1 = torch.where((theta >= 0) & (theta <= 0.5 * torch.pi), ones, zeros)
    quad2 = torch.where((theta > 0.5 * torch.pi) & (theta <= torch.pi), ones, zeros)
    quad3 = torch.where((theta > torch.pi) & (theta <= 1.5 * torch.pi), ones, zeros)
    quad4 = torch.where((theta > 1.5 * torch.pi) & (theta <= 2 * torch.pi), ones, zeros)

    a1 = 3.58396766000856
    a2 = 3.28530926421398
    a3 = 2.47465193074208
    a4 = 2.11503551932097
    b1 = -2.00351054735044
    b2 = -0.667838639957490
    b3 = -1.04946191312213
    b4 = -0.586064428952415
    lmbd = 0.8599513039

    s1 = (a1 * torch.sin(theta * lmbd) + b1 * torch.cos(theta * lmbd)) * quad1
    s2 = (a2 * torch.sin(theta * lmbd) + b2 * torch.cos(theta * lmbd)) * quad2
    s3 = (a3 * torch.sin(theta * lmbd) + b3 * torch.cos(theta * lmbd)) * quad3
    s4 = (a4 * torch.sin(theta * lmbd) + b4 * torch.cos(theta * lmbd)) * quad4
    real = s1 + s2 + s3 + s4

    plt.figure(figsize=(6, 5))
    plt.plot(theta,
             out,
             label="NN",
             )
    plt.plot(theta,
             real,
             label="REAL",
             )
    plt.title("ReCoNN phi")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ITER = 50000
    EXP = "005"
    plot_ReCoNN(ITER, EXP)
    plot_ReCoNN_phi(ITER, EXP)
