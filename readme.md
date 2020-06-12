# PyTorch implementation of "Adaptive Checkpoint Adjoint" (ACA) for an accurate and differentiable ODE solver
This library provides ordinary differential equation (ODE) solvers implemented in PyTorch. Backpropagation through all solvers is supported using the adjoint method with a checkpoint strategy to guarantee numerical accuracy in reverse-mode trajectory. For usage of ODE solvers in deep learning applications, see [1].

## Dependencies
PyTorch 1.0 (Will test on other versions later)
tensorboardX
Pythorn 3

## Examples
### Image classification on Cifar
A ResNet18 is modified into its corresponding ODE model, and achieve ~5% errorate (vs 10% by adjoint method and naive method).
Code is in folder ```cifar_classification```
#### How to train
```
python train.py

```


### Three-body problem
Please run ```python three_body_problem.py ```.
The problem is: given trajectories of three stars, how to estimate their masses and predict their future trajectory.




## References
[1] Zhuang, Juntang, et al. "Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE." arXiv preprint arXiv:2006.02493 (2020). [[arxiv]](https://arxiv.org/abs/2006.02493)

Please cite our paper if you find this repository useful:
```
@article{zhuang2020adaptive,
  title={Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE},
  author={Zhuang, Juntang and Dvornek, Nicha and Li, Xiaoxiao and Tatikonda, Sekhar and Papademetris, Xenophon and Duncan, James},
  journal={ICML},
  year={2020}
}
```
