# PyTorch implementation of "Adaptive Checkpoint Adjoint" (ACA) for an accurate and differentiable ODE solver
- This library provides ordinary differential equation (ODE) solvers implemented in PyTorch. <br/>
- Compared with ```torchdiffeq``` implementation, ACA uses a trajectory checkpoint strategy to guarantee numerical accuracy in reverse-mode trajectory, hence is more accurate in gradient estimation. <br/>
- To our knowledge, ACA is the first method to enable Neural-ODE model to outperform a standard ResNet model on benchmark such as Cifar classification, which also supports adaptive-stepsize and error estimation as the default of most widely used softwares. <br/>
- ACA also support conventional parametric ODE models. <br/>
- ACA is written in PyTorch, hence supports automatic differentiation, and can be plugged into exisiting neural network models. Furthermore, with ACA, we can build ODE models, and efficiently estimate the unkown parameters inside the model using optimizers provided by PyTorch.

## Dependencies
- PyTorch 1.0 (Will test on other versions later)
- tensorboardX
- Pythorn 3

## Examples
### Three-body problem
Please run ```python three_body_problem.py ```. <br/>
The problem is: given trajectories of three stars, how to estimate their masses and predict their future trajectory.<br/>
[Watch the videos in folde ```figures```](https://www.youtube.com/playlist?list=PL7KkG3n9bER4ODAMzAKzfXIaF0ndUxK-N)
[![Alt text](./figures/three_body.png)](https://www.youtube.com/playlist?list=PL7KkG3n9bER4ODAMzAKzfXIaF0ndUxK-N)

### Image classification on Cifar
A ResNet18 is modified into its corresponding ODE model, and achieve ~5% errorate (vs 10% by adjoint method and naive method).
Code is in folder ```cifar_classification```
#### How to train
```
python train.py
```
You can visualize the training and validation curve with 
```
tensorboard --logdir cifar_classification/resnet/resnet_RK12_lr_0.1_h_None
```

#### Train with different modes of solvers
- End-time fast mode <br/>
```train.py``` uses the solver defined in ```torch_ACA/odesolver_mem/ode_solver_endtime.py```, this mode only support integration from start time t0 to end time t1, and output a tensor for time t1.

- End-time memory-efficient mode <br/>
```train_mem.py``` uses the solver defined in ```torch_ACA/odesolver_mem/adjoint_mem.py```, this mode only support integration from start time t0 to end time t1, and output a tensor for time t1. Furtheremore, this mode uses O(Nf + Nt) memory, which is more memory-efficient than normal mode, but the running time is longer.

- Multiple evaluation time-points mode <br/>
```train_multieval.py``` uses the solver defined in ```torch_ACA/odesolver/ode_solver.py```, this mode supports extracting outputs from multiple time points between t0 and t1. 

- Note for multiple evaluation time-points mode: <br/>
```
   (1) Evaluation time 't_eval' must be specified in a list. 
        e.g.  t_eval = [a1, a2, a3 ..., an]  where t0 < a1 < a2 < ... t1, or t1 < a1 < a2 < ... < t0 
   (2) suppose 'z' is of shape 'AxBxCx...', then the output is of shape 'nxAxBxCx...' 
```

#### Warning
- This repository currently only supports ``` \frac{dz}{dt} = f(t,z) ``` where ```z``` is a tensor (other data types such as tuple are not supported). <br/>
- If you are using a function ```f``` which produces many output tensors or ```z``` is a list of tensors, you can concatenate them into a single tensor within definition of ```f```.

### Results
<img src="./figures/results.png">

## References
[1] Zhuang, Juntang, et al. "Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE." arXiv preprint arXiv:2006.02493 (2020). [[arxiv]](https://arxiv.org/abs/2006.02493) <br/>

Please cite our paper if you find this repository useful:
```
@article{zhuang2020adaptive,
  title={Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE},
  author={Zhuang, Juntang and Dvornek, Nicha and Li, Xiaoxiao and Tatikonda, Sekhar and Papademetris, Xenophon and Duncan, James},
  journal={ICML},
  year={2020}
}
```
