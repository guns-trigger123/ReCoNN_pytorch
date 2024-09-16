import os
import torch.optim as optim
from torch import nn
from utils import *


class MLP_2D_InteriorMaterial(nn.Module):
    def __init__(self):
        super(MLP_2D_InteriorMaterial, self).__init__()
        self.fcn_w = nn.Sequential(
            nn.Linear(2, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 6),
        )
        self.fcn_phi = nn.Sequential(
            nn.Linear(2, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 3),
        )
        self.lmbd = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, x):
        r = torch.norm(x, p=2, dim=1, keepdim=True)
        x_ba = x / r
        eta = self._eta(r)
        w = x
        for layer in self.fcn_w:
            w = layer(w)
        phi = x_ba
        for layer in self.fcn_phi:
            phi = layer(phi)
        rp = w[:, 0:1] + w[:, 3:4] * eta + (w[:, 1:2] + w[:, 4:5] * eta) * torch.abs(x[:, 0:1]) + (w[:, 2:3] + w[:, 5:6] * eta) * torch.abs(x[:, 1:2])
        sp = (phi[:, 0:1] + phi[:, 1:2] * torch.abs(x_ba[:, 0:1]) + phi[:, 2:3] * torch.abs(x_ba[:, 1:2])) * eta * (r ** self.lmbd[0])
        return rp + sp

    def phi(self, x_ba):
        for layer in self.fcn_phi:
            x_ba = layer(x_ba)
        return x_ba

    def w(self, x):
        r = torch.norm(x, p=2, dim=1, keepdim=True)
        eta = self._eta(r)
        for layer in self.fcn_w:
            x = layer(x)
        return x[:, 0:1] + x[:, 3:4] * eta, x[:, 1:2] + x[:, 4:5] * eta, x[:, 2:3] + x[:, 5:6] * eta

    @staticmethod
    def _eta(r):
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
    a1, a2, a3, a4 = 3.58396766000856, 3.28530926421398, 2.47465193074208, 2.11503551932097
    b1, b2, b3, b4 = -2.00351054735044, -0.667838639957490, -1.04946191312213, -0.586064428952415
    lmbd = 0.8599513039

    r = torch.norm(x, p=2, dim=1, keepdim=True)
    theta = - torch.arctan(x[:, 0:1] / x[:, 1:2])
    pi_add = torch.where(x[:, 1:2] < 0, torch.pi * torch.ones_like(theta, device=theta.device), torch.zeros_like(theta, device=theta.device))
    theta = lmbd * (theta + pi_add + torch.pi / 2)
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


def pde_weight(x):
    ones = torch.ones_like(x[:, 0:1], device=x.device)
    x_sqr = 40 * (torch.norm(x, p=2, dim=1, keepdim=True) ** 2).detach()
    return torch.where(x_sqr < 1, x_sqr, ones)


def int_weight(x):
    ones = torch.ones_like(x[:, 0:1], device=x.device)
    x_abs = 40 * torch.norm(x, p=2, dim=1, keepdim=True).detach()
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
    else:
        x_int = None
    return x_int


if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP_2D_InteriorMaterial()
    model.to(device)
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ExponentialLR(opt, gamma=1e-3 ** (1 / 25000))

    x = domain(250).to(device).requires_grad_()
    sig, src, pde_w = sigma(x), source(x), pde_weight(x)

    x_bc = bc(250).to(device)
    zeros_bc = torch.zeros_like(x_bc[:, 0:1], device=x_bc.device)

    x_int, gv_x_int, gnv_x_int, nv_x_int, int_w = [], [], [], [], []
    for i, index in enumerate([0, 1, 0, 1]):
        x_int.append(interface(250, i).to(device).requires_grad_())
        gv_x_int.append(gradient(x_int[i][:, index:index + 1], x_int[i]).detach())
        gnv_x_int.append(torch.norm(gv_x_int[i], p=2, dim=1, keepdim=True))
        nv_x_int.append(gv_x_int[i] / gnv_x_int[i])
        int_w.append(int_weight(x_int[i]))

    x_phi, gv_x_phi, gnv_x_phi, nv_x_phi = [], [], [], []
    for i, (value, index) in enumerate(zip([(0.0, 0.0125), (-0.0125, 0.0), (0.0, -0.0125), (0.0125, 0.0)], [0, 1, 0, 1])):
        x_phi.append(torch.tensor([value], device=device).requires_grad_())
        gv_x_phi.append(gradient(x_phi[i][:, index:index + 1], x_phi[i]).detach())
        gnv_x_phi.append(torch.norm(gv_x_phi[i], p=2, dim=1, keepdim=True))
        nv_x_phi.append(gv_x_phi[i] / gnv_x_phi[i])

    for iter in range(50000):
        u_x = model(x)
        lap_u_x = laplace(u_x, x)
        loss_pde = torch.mean(pde_w * ((lap_u_x * sig - src) ** 2))

        u_bc = model(x_bc)
        loss_bc = criterion(u_bc, zeros_bc)

        # loss_inty = []
        # for i, sigm in zip([0, 2], [(1, 2), (4, 3)]):
        #     r_int = torch.norm(x_int[i], p=2, dim=1, keepdim=True)
        #     eta_int, x_ba_int = model._eta(r_int), x_int[i] / r_int
        #     phi_int = model.phi(x_ba_int)
        #     phi_int0 = phi_int[:, 0:1] * eta_int * (r_int ** model.lmbd[0])
        #     phi_int1 = phi_int[:, 1:2] * eta_int * (r_int ** (model.lmbd[0] - 1.0))
        #     phi_int2 = phi_int[:, 2:3] * eta_int * (r_int ** model.lmbd[0])
        #     absv_ba_int = torch.abs(x_ba_int[:, 1:2])
        #     w_int0, w_int1, w_int2 = model.w(x_int[i])
        #     absv_int, bc_cut = torch.abs(x_int[i][:, 1:2]), torch.ones_like(x_int[i][:, 0:1], device=x_int[i].device)
        #     grad_term = gradient(phi_int0 + phi_int2 * absv_ba_int + bc_cut * (w_int0 + w_int2 * absv_int), x_int[i])
        #     dire_deri_term = torch.sum(grad_term * nv_x_int[i], dim=1, keepdim=True)
        #     dleft, dright = dire_deri_term - (w_int1 * bc_cut + phi_int1), dire_deri_term + (w_int1 * bc_cut + phi_int1)
        #     sleft, sright = sigm[1], sigm[0]
        #     loss_inty.append(torch.mean((sleft * dleft - sright * dright) ** 2 * int_w[i]))
        # loss_intx = []
        # for i, sigm in zip([1, 3], [(2, 3), (1, 4)]):
        #     r_int = torch.norm(x_int[i], p=2, dim=1, keepdim=True)
        #     eta_int, x_ba_int = model._eta(r_int), x_int[i] / r_int
        #     phi_int = model.phi(x_ba_int)
        #     phi_int0 = phi_int[:, 0:1] * eta_int * (r_int ** model.lmbd[0])
        #     phi_int1 = phi_int[:, 1:2] * eta_int * (r_int ** model.lmbd[0])
        #     phi_int2 = phi_int[:, 2:3] * eta_int * (r_int ** (model.lmbd[0] - 1.0))
        #     absv_ba_int = torch.abs(x_ba_int[:, 0:1])
        #     w_int0, w_int1, w_int2 = model.w(x_int[i])
        #     absv_int, bc_cut = torch.abs(x_int[i][:, 0:1]), torch.ones_like(x_int[i][:, 0:1], device=x_int[i].device)
        #     grad_term = gradient(phi_int0 + phi_int1 * absv_ba_int + bc_cut * (w_int0 + w_int1 * absv_int), x_int[i])
        #     dire_deri_term = torch.sum(grad_term * nv_x_int[i], dim=1, keepdim=True)
        #     dbottom, dtop = dire_deri_term - (w_int2 * bc_cut + phi_int2), dire_deri_term + (w_int2 * bc_cut + phi_int2)
        #     sbottom, stop = sigm[1], sigm[0]
        #     loss_intx.append(torch.mean((sbottom * dbottom - stop * dtop) ** 2 * int_w[i]))
        # loss_int = loss_inty[0] + loss_inty[1] + loss_intx[0] + loss_intx[1]

        loss_inty = []
        for i, sigm in zip([0, 2], [(1, 2), (4, 3)]):
            w_int0, w_int1, w_int2 = model.w(x_int[i])
            absv_int, bc_cut = torch.abs(x_int[i][:, 1:2]), torch.ones_like(x_int[i][:, 0:1], device=x_int[i].device)
            grad_term = gradient(bc_cut * (w_int0 + w_int2 * absv_int), x_int[i])
            dire_deri_term = torch.sum(grad_term * nv_x_int[i], dim=1, keepdim=True)
            dleft, dright = dire_deri_term - (w_int1 * bc_cut), dire_deri_term + (w_int1 * bc_cut)
            sleft, sright = sigm[1], sigm[0]
            loss_inty.append(torch.mean((sleft * dleft - sright * dright) ** 2 * int_w[i]))
        loss_intx = []
        for i, sigm in zip([1, 3], [(2, 3), (1, 4)]):
            w_int0, w_int1, w_int2 = model.w(x_int[i])
            absv_int, bc_cut = torch.abs(x_int[i][:, 0:1]), torch.ones_like(x_int[i][:, 0:1], device=x_int[i].device)
            grad_term = gradient(bc_cut * (w_int0 + w_int1 * absv_int), x_int[i])
            dire_deri_term = torch.sum(grad_term * nv_x_int[i], dim=1, keepdim=True)
            dbottom, dtop = dire_deri_term - (w_int2 * bc_cut), dire_deri_term + (w_int2 * bc_cut)
            sbottom, stop = sigm[1], sigm[0]
            loss_intx.append(torch.mean((sbottom * dbottom - stop * dtop) ** 2 * int_w[i]))
        loss_int = loss_inty[0] + loss_inty[1] + loss_intx[0] + loss_intx[1]

        # loss_phiy = []
        # for i, sigm in zip([0, 2], [(1, 2), (4, 3)]):
        #     r_phi = torch.norm(x_phi[i], p=2, dim=1, keepdim=True)
        #     eta_phi, x_ba_phi = model._eta(r_phi), x_phi[i] / r_phi
        #     phi_phi = model.phi(x_ba_phi)
        #     phi_phi0 = phi_phi[:, 0:1] * eta_phi * (r_phi ** model.lmbd[0])
        #     phi_phi1 = phi_phi[:, 1:2] * eta_phi * (r_phi ** (model.lmbd[0] - 1.0))
        #     phi_phi2 = phi_phi[:, 2:3] * eta_phi * (r_phi ** model.lmbd[0])
        #     absv_ba_phi = torch.abs(x_ba_phi[:, 1:2])
        #     grad_term = gradient(phi_phi0 + phi_phi2 * absv_ba_phi, x_phi[i])
        #     dire_deri_term = torch.sum(grad_term * nv_x_phi[i], dim=1, keepdim=True)
        #     dleft, dright = dire_deri_term - phi_phi1, dire_deri_term + phi_phi1
        #     sleft, sright = sigm[1], sigm[0]
        #     loss_phiy.append(torch.mean((sleft * dleft - sright * dright) ** 2))
        # loss_phix = []
        # for i, sigm in zip([1, 3], [(2, 3), (1, 4)]):
        #     r_phi = torch.norm(x_phi[i], p=2, dim=1, keepdim=True)
        #     eta_phi, x_ba_phi = model._eta(r_phi), x_phi[i] / r_phi
        #     phi_phi = model.phi(x_ba_phi)
        #     phi_phi0 = phi_phi[:, 0:1] * eta_phi * (r_phi ** model.lmbd[0])
        #     phi_phi1 = phi_phi[:, 1:2] * eta_phi * (r_phi ** model.lmbd[0])
        #     phi_phi2 = phi_phi[:, 2:3] * eta_phi * (r_phi ** (model.lmbd[0] - 1.0))
        #     absv_ba_phi = torch.abs(x_ba_phi[:, 0:1])
        #     grad_term = gradient(phi_phi0 + phi_phi1 * absv_ba_phi, x_phi[i])
        #     dire_deri_term = torch.sum(grad_term * nv_x_phi[i], dim=1, keepdim=True)
        #     dbottom, dtop = dire_deri_term - phi_phi2, dire_deri_term + phi_phi2
        #     sbottom, stop = sigm[1], sigm[0]
        #     loss_phix.append(torch.mean((sbottom * dbottom - stop * dtop) ** 2))
        # loss_phi = loss_phiy[0] + loss_phiy[1] + loss_phix[0] + loss_phix[1]

        loss_phiy = []
        for i, sigm in zip([0, 2], [(1, 2), (4, 3)]):
            r_phi = torch.norm(x_phi[i], p=2, dim=1, keepdim=True)
            x_ba_phi = x_phi[i] / r_phi
            phi_phi = model.phi(x_ba_phi)
            phi_phi0 = phi_phi[:, 0:1]
            phi_phi1 = phi_phi[:, 1:2]
            phi_phi2 = phi_phi[:, 2:3]
            absv_ba_phi = torch.abs(x_ba_phi[:, 1:2])
            grad_term = gradient(phi_phi0 + phi_phi2 * absv_ba_phi, x_ba_phi)
            dire_deri_term = torch.sum(grad_term * nv_x_phi[i], dim=1, keepdim=True)
            dleft, dright = dire_deri_term - phi_phi1, dire_deri_term + phi_phi1
            sleft, sright = sigm[1], sigm[0]
            loss_phiy.append(torch.mean((sleft * dleft - sright * dright) ** 2))
        loss_phix = []
        for i, sigm in zip([1, 3], [(2, 3), (1, 4)]):
            r_phi = torch.norm(x_phi[i], p=2, dim=1, keepdim=True)
            x_ba_phi = x_phi[i] / r_phi
            phi_phi = model.phi(x_ba_phi)
            phi_phi0 = phi_phi[:, 0:1]
            phi_phi1 = phi_phi[:, 1:2]
            phi_phi2 = phi_phi[:, 2:3]
            absv_ba_phi = torch.abs(x_ba_phi[:, 0:1])
            grad_term = gradient(phi_phi0 + phi_phi1 * absv_ba_phi, x_ba_phi)
            dire_deri_term = torch.sum(grad_term * nv_x_phi[i], dim=1, keepdim=True)
            dbottom, dtop = dire_deri_term - phi_phi2, dire_deri_term + phi_phi2
            sbottom, stop = sigm[1], sigm[0]
            loss_phix.append(torch.mean((sbottom * dbottom - stop * dtop) ** 2))
        loss_phi = loss_phiy[0] + loss_phiy[1] + loss_phix[0] + loss_phix[1]

        loss = torch.sqrt(loss_pde) + 3.1623 * torch.sqrt(loss_int) + 10 * torch.sqrt(loss_bc) + 1 * torch.sqrt(loss_phi)

        loss.backward()
        opt.step()
        if iter > 24999:
            sch.step()
        model.zero_grad()
        x.grad = None
        for xi in x_int:
            xi.grad = None
        for xi in x_phi:
            xi.grad = None

        if (iter + 1) % 1000 == 0:
            x = domain(250).to(device).requires_grad_()
            sig, src, pde_w = sigma(x), source(x), pde_weight(x)

            x_bc = bc(250).to(device)
            zeros_bc = torch.zeros_like(x_bc[:, 0:1], device=x_bc.device)

            x_int, gv_x_int, gnv_x_int, nv_x_int, int_w = [], [], [], [], []
            for i, index in enumerate([0, 1, 0, 1]):
                x_int.append(interface(250, i).to(device).requires_grad_())
                gv_x_int.append(gradient(x_int[i][:, index:index + 1], x_int[i]).detach())
                gnv_x_int.append(torch.norm(gv_x_int[i], p=2, dim=1, keepdim=True))
                nv_x_int.append(gv_x_int[i] / gnv_x_int[i])
                int_w.append(int_weight(x_int[i]))

            x_phi, gv_x_phi, gnv_x_phi, nv_x_phi = [], [], [], []
            for i, (value, index) in enumerate(zip([(0.0, 0.0125), (-0.0125, 0.0), (0.0, -0.0125), (0.0125, 0.0)], [0, 1, 0, 1])):
                x_phi.append(torch.tensor([value], device=device).requires_grad_())
                gv_x_phi.append(gradient(x_phi[i][:, index:index + 1], x_phi[i]).detach())
                gnv_x_phi.append(torch.norm(gv_x_phi[i], p=2, dim=1, keepdim=True))
                nv_x_phi.append(gv_x_phi[i] / gnv_x_phi[i])

        if (iter + 1) % 100 == 0:
            print(f"iter: {iter} " +
                  f"loss: {loss} loss_pde: {loss_pde} loss_bc: {loss_bc} loss_int: {loss_int} loss_phi: {loss_phi}")

        if (iter + 1) % 500 == 0:
            save_path = os.path.join('../saved_models/2D_material/experiment 3/', f'ReCoNN_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)
