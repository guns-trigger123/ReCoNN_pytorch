import matplotlib.pyplot as plt
from utils import *
from train_ReCoNN_2D_smooth import MLP_2D_SMOOTH


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


def plot_ReCoNN():
    model = MLP_2D_SMOOTH([varphi])
    model.load_state_dict(torch.load(("../saved_models/2D_smooth/ReCoNN_50000.pt")))

    x, y = torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256)
    input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    input_x, input_y = input_x.reshape(-1, 1), input_y.reshape(-1, 1)
    input = torch.cat([input_x, input_y], dim=1)

    out = model(input).detach()
    plt.figure(figsize=(6, 5))
    plt.scatter(input[:, 0:1],
                input[:, 1:2],
                s=2,
                # c=out,
                # c=real(input),
                c=torch.abs(out - real(input)),
                cmap='rainbow'
                )
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.colorbar()
    # plt.title("ReCoNN")
    plt.title("error")
    # plt.title("real")
    plt.show()


if __name__ == '__main__':
    plot_ReCoNN()
