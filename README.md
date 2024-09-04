# Regularity-Conforming Neural Networks (ReCoNNs) for solving Partial Differential Equations

This is the **unofficial** pytorch version implementation of the paper "Regularity-Conforming Neural Networks (ReCoNNs)
for
solving Partial Differential Equations".

## Numerical results

### 1 1D Transmission problem

Compare the results of "vanilla PINNs[[1]](https://doi.org/10.1016/j.jcp.2018.10.045)" and "
XPINNs[[2]](https://doi.org/10.4208/cicp.OA-2020-0164)" with those of "ReCoNNs".

![PINN](./assets/1D_case/PINN.png)
![xPINN](./assets/1D_case/xPINN.png)
![ReCoNN](./assets/1D_case/ReCoNN.png)

## Citation

If you find our work useful in your research, please cite:

```
@article{taylor2024regularity,
  title={Regularity-Conforming Neural Networks (ReCoNNs) 
         for solving Partial Differential Equations},
  author={Taylor, Jamie M 
          and Pardo, David 
          and Mu{\~n}oz-Matute, Judit},
  journal={arXiv preprint arXiv:2405.14110},
  year={2024}
}
```

## Other reference

[1] Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving
forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational physics,
2019, 378: 686-707.

[2] Jagtap A D, Karniadakis G E. Extended physics-informed neural networks (XPINNs): A generalized space-time domain
decomposition based deep learning framework for nonlinear partial differential equations[J]. Communications in
Computational Physics, 2020, 28(5).