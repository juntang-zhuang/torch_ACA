#!/usr/bin/env python
# coding: utf-8

# In[27]:
#pydevd.settrace(suspend=False, trace_only_current_thread=True)

# Import scipy
import scipy as sci
# Import matplotlib and associated modules for 3D and animations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from torch_ACA.odesolver import odesolve
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from torchdiffeq import odeint_adjoint as odeint
import pandas as pd
import shutil
# In[28]:

animation_fig_path = './figures_animation'
save_fig_path = './figures_epoch_three_body'

if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

if not os.path.exists(animation_fig_path):
    os.makedirs(animation_fig_path)


num_epochs = 100
lr = 1e-1
lr_decay = 0.95
loss_weight_decay = 1e-2

set_legend = False
TrainMode = True
resume = None#'{}/best_model.pth'.format(save_fig_path)

time_span = sci.linspace(0, 1.0, 1000)  # 20 orbital periods and 500 points

# Define universal gravitation constant
G = 6.67408e-11  # N-m2/kg2
# Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri
# Net constants
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define masses
m1 = 1.1  # Alpha Centauri A
m2 = 1.1  # Alpha Centauri B
m3 = 0.90  # Third Star


'''
# Define initial position vectors
r1 = [0.5, 0, 0]  # m
r2 = [0, 0.5, 0.5]  # m
r3 = [0, 0., 0.5]  # m

# Convert pos vectors to arrays
r1 = sci.array(r1, dtype="float64")
r2 = sci.array(r2, dtype="float64")
# Find Centre of Mass
r_com = (m1 * r1 + m2 * r2) / (m1 + m2)
# Define initial velocities
v1 = [0.0, 0.1, 0]  # m/s
v2 = [-0.0, 0, 0.1]  # m/s
v3 = [ 0.1, 0, 0]
'''


r1 = [0.0, 0.2, 0]  # m
r2 = [0.5, 0, 0.5]  # m
r3 = [0, 0., 0.5]  # m

# Convert pos vectors to arrays
r1 = sci.array(r1, dtype="float64")
r2 = sci.array(r2, dtype="float64")
# Find Centre of Mass
r_com = (m1 * r1 + m2 * r2) / (m1 + m2)
# Define initial velocities
v1 = [0.0, 0.1, 0.0]  # m/s
v2 = [-0.0, 0, 0.1]  # m/s
v3 = [ 0.1, 0, 0]


# Convert velocity vectors to arrays
v1 = sci.array(v1, dtype="float64")
v2 = sci.array(v2, dtype="float64")
# Mass of the Third Star
# Position of the Third Star
r3 = sci.array(r3, dtype="float64")
# Velocity of the Third Star
v3 = sci.array(v3, dtype="float64")

# Update COM formula
r_com = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)
# Update velocity of COM formula
v_com = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)


def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
    dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    r12_derivs = sci.concatenate((dr1bydt, dr2bydt))
    r_derivs = sci.concatenate((r12_derivs, dr3bydt))
    v12_derivs = sci.concatenate((dv1bydt, dv2bydt))
    v_derivs = sci.concatenate((v12_derivs, dv3bydt))
    derivs = sci.concatenate((r_derivs, v_derivs))
    return derivs


# Package initial parameters
init_params = sci.array([r1, r2, r3, v1, v2, v3])  # Initial parameters
init_params = init_params.flatten()  # Flatten to make 1D array

# Run the ODE solver
import scipy.integrate

three_body_sol = sci.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3), rtol=1e-7, atol=1e-7)

r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]


# plot
# Create figure
fig = plt.figure(figsize=(15, 15))
# Create 3D axes
ax = fig.add_subplot(111, projection="3d")
# Plot the orbits
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2])
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2])
ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2])

# Plot the final positions of the stars
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], marker="o", s=100, label="Alpha Centauri A")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], marker="o", s=100, label="Alpha Centauri B")
ax.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], marker="o", s=100, label="Alpha Centauri C")

# Add a few more bells and whistles
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
ax.legend(loc="upper left", fontsize=14)

plt.draw()
plt.pause(0.001)
plt.savefig('%s/truth.png' % save_fig_path)

# use pytorch ode solver to estimate mass from trajectory observations
# define function as a PyTorch module
class TorchThreeBodyEquations(nn.Module):
    def __init__(self):
        super(TorchThreeBodyEquations, self).__init__()
        self.G = G
        self.m1 = nn.Parameter((torch.rand(1) - 0.5)* 0.20 + 1.0)
        self.m2 = nn.Parameter((torch.rand(1) - 0.5)* 0.20 + 1.0)
        self.m3 = nn.Parameter((torch.rand(1) - 0.5)* 0.20 + 1.0)

    def forward(self, t, w):
        r1 = w[...,:3]
        r2 = w[...,3:6]
        r3 = w[...,6:9]
        v1 = w[...,9:12]
        v2 = w[...,12:15]
        v3 = w[...,15:18]
        r12 = torch.norm(r2 - r1)
        r13 = torch.norm(r3 - r1)
        r23 = torch.norm(r3 - r2)
        dv1bydt = K1 * self.m2 * (r2 - r1) / r12 ** 3 + K1 * self.m3 * (r3 - r1) / r13 ** 3
        dv2bydt = K1 * self.m1 * (r1 - r2) / r12 ** 3 + K1 * self.m3 * (r3 - r2) / r23 ** 3
        dv3bydt = K1 * self.m1 * (r1 - r3) / r13 ** 3 + K1 * self.m2 * (r2 - r3) / r23 ** 3
        dr1bydt = K2 * v1
        dr2bydt = K2 * v2
        dr3bydt = K2 * v3
        derivs = torch.cat((dr1bydt, dr2bydt,dr3bydt, dv1bydt, dv2bydt, dv3bydt), dim=-1)
        return derivs


# create training data
trajectory = torch.from_numpy(np.concatenate((r1_sol, r2_sol, r3_sol), -1)).float()
trajectory = torch.unsqueeze(trajectory, 0)

func = TorchThreeBodyEquations()
t_list = time_span.tolist()

initial_condition = torch.from_numpy(np.concatenate((r1, r2, r3, v1, v2, v3), -1)).float()
initial_condition = torch.unsqueeze(initial_condition, 0)


if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

# configure training options
options = {}
options.update({'method': 'Dopri5'})
options.update({'h': None})
options.update({'t0': 0.0})
options.update({'t1': 1.0})
options.update({'rtol': 1e-7})
options.update({'atol': 1e-7})
options.update({'print_neval': False})
options.update({'neval_max': 1000000})
options.update({'safety': None})
options.update({'t_eval':t_list})

optimizer = torch.optim.Adam(func.parameters(), lr=lr, betas=(0.50, 0.50))

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        lr_old = param_group['lr']
        param_group['lr'] = lr

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth.tar'))

func.train()

if resume is not None:
    func.load_state_dict(torch.load(resume))

# Create figure
fig = plt.figure(figsize=(15, 5))

loss_weight_decay = loss_weight_decay ** time_span
loss_weight_decay = torch.from_numpy(loss_weight_decay).float()

loss_history = []

best_loss = np.inf
if TrainMode:
    for _epoch in range(num_epochs):
        print('M1 {}, estimated m1 {}'.format(m1, func.m1.item()))
        print('M2 {}, estimated m2 {}'.format(m2, func.m2.item()))
        print('M3 {}, estimated m3 {}'.format(m3, func.m3.item()))

        lr *= lr_decay
        adjust_learning_rate(optimizer, lr)
        optimizer.zero_grad()


        func.eval()

        out = odesolve(func, initial_condition, options=options)#, time_points=t_list)
        out_all2 = out.permute(1,0,-1)

        '''
        solver = Dopri5(func, t0=options['t0'],t1=options['t1'],h=options['h'],
                      rtol =options['rtol'],atol=options['atol'],neval_max=options['neval_max'],
                      safety=options['safety'], keep_small_step=options['keep_small_step'])
        out2 = solver.integrate(initial_condition, t_eval = t_list)
        out_all2 = out2.permute(1,0,-1)
        '''

        out_all_data = out_all2.data.cpu().numpy()
        position = out_all2[..., :9]
        # loss = torch.sum((position - trajectory) ** 2)

        dif = position - trajectory
        dif = torch.sum(dif**2, -1, keepdim=False) # 1 x N
        dif = torch.squeeze(dif) # N
        dif2 = dif * loss_weight_decay
        loss = torch.sum(torch.abs(dif2)) / float(time_span.shape[0])

        if torch.sum(dif).item() < best_loss:
            best_loss = torch.sum(dif).item()
            save_checkpoint(func.state_dict(), True, checkpoint=save_fig_path, filename='best_model.pth')

        loss_history.append(torch.sum(dif).item())

        loss.backward()

        optimizer.step()
        print('Epoch %d: Loss: %.8f' % (_epoch, loss.item()))

        plt.clf()
        # Create 3D axes
        ax = fig.add_subplot(131, projection="3d")
        # Plot the orbits
        ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], label='Ground truth')
        ax.plot(out_all_data[0, :, 0], out_all_data[0, :, 1], out_all_data[0, :, 2], label='Fitting')
        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlabel("z-coordinate", fontsize=14)
        # ax.set_title("Star A trajectory", fontsize=14)
        ax.legend()#loc="upper left", fontsize=14)

        # Create 3D axes
        ax = fig.add_subplot(132, projection="3d")
        # Plot the orbits
        ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], label='Ground truth')
        ax.plot(out_all_data[0, :, 3], out_all_data[0, :, 4], out_all_data[0, :, 5], label='Fitting')
        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlabel("z-coordinate", fontsize=14)
        # ax.set_title("Star B trajectory", fontsize=14)
        ax.legend()#loc="upper left", fontsize=14)

        # Create 3D axes
        ax = fig.add_subplot(133, projection="3d")
        # Plot the orbits
        ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], label='Ground truth')
        ax.plot(out_all_data[0, :, 6], out_all_data[0, :, 7], out_all_data[0, :, 8], label='Fitting')
        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlabel("z-coordinate", fontsize=14)
        # ax.set_title("Star C trajectory", fontsize=14)
        ax.legend()#loc="upper left", fontsize=14)

        plt.savefig('%s/curve_epoch%d.png' % (save_fig_path, _epoch))
        plt.draw()
        plt.pause(0.001)


    # ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="tab:red")

    print('Finished training')
    df = pd.DataFrame(np.array(loss_history))
    df.to_csv(save_fig_path+'/three_body.csv')

#################################################################################################
# load best model, and generate curves
# use time points which are unseen in the training data
#func.load_state_dict(torch.load(resume))
time_span = sci.linspace(0, 2.0, 2000)  # 20 orbital periods and 500 points
time_span = time_span + time_span[1]/2.0
time_span[0] = 0.0
t_list = time_span.tolist()
options.update({'t_eval':t_list})

three_body_sol = sci.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3), rtol=1e-7, atol=1e-7)

r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]

# create training data
trajectory = torch.from_numpy(np.concatenate((r1_sol, r2_sol, r3_sol), -1)).float()
trajectory = torch.unsqueeze(trajectory, 0)

loss_weight_decay = 1e-1
loss_weight_decay = loss_weight_decay ** time_span
loss_weight_decay = torch.from_numpy(loss_weight_decay).float()

def test(func, save_fig_path):
    func.load_state_dict(torch.load( save_fig_path +'/best_model.pth' ))

    '''
    solver = Dopri5(func, t0=options['t0'], t1=options['t1'], h=options['h'],
                  rtol=options['rtol'], atol=options['atol'], neval_max=options['neval_max'],
                  safety=options['safety'], keep_small_step=options['keep_small_step'])
    out = solver.integrate(initial_condition, t_eval=t_list)
    '''

    out = odesolve(func, initial_condition, options=options)  # , time_points=t_list)
    out_all = out.permute(1, 0, -1)

    out_all_data = out_all.data.cpu().numpy()
    position = out_all[..., :9]

    plt.clf()
    # Create 3D axes
    ax = fig.add_subplot(131, projection="3d")
    # Plot the orbits
    ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], '-',label='Ground truth', linewidth=2.0)
    ax.plot(out_all_data[0, :, 0], out_all_data[0, :, 1], out_all_data[0, :, 2], '--',label='Fitting', linewidth=2.0)
    if set_legend:
        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlabel("z-coordinate", fontsize=14)
        # ax.set_title("Star A trajectory", fontsize=14)
        ax.legend()#loc="upper left", fontsize=14)

    # Create 3D axes
    ax = fig.add_subplot(132, projection="3d")
    # Plot the orbits
    ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], '-',label='Ground truth', linewidth=2.0)
    ax.plot(out_all_data[0, :, 3], out_all_data[0, :, 4], out_all_data[0, :, 5], '--',label='Fitting', linewidth=2.0)
    if set_legend:
        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlabel("z-coordinate", fontsize=14)
        # ax.set_title("Star B trajectory", fontsize=14)
        ax.legend()#loc="upper left", fontsize=14)

    # Create 3D axes
    ax = fig.add_subplot(133, projection="3d")
    # Plot the orbits
    ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], '-',label='Ground truth', linewidth=2.0)
    ax.plot(out_all_data[0, :, 6], out_all_data[0, :, 7], out_all_data[0, :, 8], '--',label='Fitting', linewidth=2.0)
    if set_legend:
        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlabel("z-coordinate", fontsize=14)
        # ax.set_title("Star C trajectory", fontsize=14)
        ax.legend()#loc="upper left", fontsize=14)

    plt.savefig('%s/best_model.png' % (save_fig_path))
    plt.draw()
    plt.pause(0.001)

    dif = position - trajectory
    dif = torch.sum(dif ** 2, -1, keepdim=False)  # 1 x N
    dif1 = torch.squeeze(dif)  # N
    dif1 = dif1 * loss_weight_decay
    loss = torch.sum(torch.abs(dif1)) / float(time_span.shape[0])

    ###################################################################
    #                  plot into animation                            #
    ###################################################################
    generate_animation_figures(out_all_data[0, :, :3], out_all_data[0, :, 3:6], out_all_data[0, :, 6:9], r1_sol, r2_sol,
                               r3_sol,
                               np.array(t_list), animation_fig_path)

    return torch.sum(dif).item()/ float(time_span.shape[0]),loss.item()

def generate_animation_figures(r1_sol,r2_sol, r3_sol, r1_truth, r2_truth, r3_truth,time_span,anima_fig_path=None):
    fig = plt.figure(figsize=(30, 15))
    for i in range(time_span.shape[0]):
        # plot
        # Create figure

        plt.clf()
        # Create 3D axes
        ax = fig.add_subplot(121, projection="3d")

        # Plot the orbits
        ax.plot(r1_sol[:i, 0], r1_sol[:i, 1], r1_sol[:i, 2])
        ax.plot(r2_sol[:i, 0], r2_sol[:i, 1], r2_sol[:i, 2])
        ax.plot(r3_sol[:i, 0], r3_sol[:i, 1], r3_sol[:i, 2])

        # Plot the final positions of the stars
        ax.scatter(r1_sol[i, 0], r1_sol[i, 1], r1_sol[i, 2], marker="o", s=100, label="Alpha Centauri A", linewidth=3.0)
        ax.scatter(r2_sol[i, 0], r2_sol[i, 1], r2_sol[i, 2], marker="o", s=100, label="Alpha Centauri B", linewidth=3.0)
        ax.scatter(r3_sol[i, 0], r3_sol[i, 1], r3_sol[i, 2], marker="o", s=100, label="Alpha Centauri C", linewidth=3.0)

        # Add a few more bells and whistles
        ax.set_xlabel("x-coordinate", fontsize=20)
        ax.set_ylabel("y-coordinate", fontsize=20)
        ax.set_zlabel("z-coordinate", fontsize=20)
        ax.set_title("Predicted Trajectory\n", fontsize=20)
        ax.legend(loc="upper left", fontsize=20)

        # plot ground truth
        ax = fig.add_subplot(122, projection="3d")

        # Plot the orbits
        ax.plot(r1_truth[:i, 0], r1_truth[:i, 1], r1_truth[:i, 2])
        ax.plot(r2_truth[:i, 0], r2_truth[:i, 1], r2_truth[:i, 2])
        ax.plot(r3_truth[:i, 0], r3_truth[:i, 1], r3_truth[:i, 2])

        # Plot the final positions of the stars
        ax.scatter(r1_truth[i, 0], r1_truth[i, 1], r1_truth[i, 2], marker="o", s=100, label="Alpha Centauri A", linewidth=3.0)
        ax.scatter(r2_truth[i, 0], r2_truth[i, 1], r2_truth[i, 2], marker="o", s=100, label="Alpha Centauri B", linewidth=3.0)
        ax.scatter(r3_truth[i, 0], r3_truth[i, 1], r3_truth[i, 2], marker="o", s=100, label="Alpha Centauri C", linewidth=3.0)

        # Add a few more bells and whistles
        ax.set_xlabel("x-coordinate", fontsize=20)
        ax.set_ylabel("y-coordinate", fontsize=20)
        ax.set_zlabel("z-coordinate", fontsize=20)
        ax.set_title("Ground Truth Trajectory\n", fontsize=20)
        ax.legend(loc="upper left", fontsize=20)

        plt.draw()
        #plt.pause(0.001)
        plt.savefig('%s/truth_step_%d.png' % (anima_fig_path, i))

best_dif, best_loss = test(func, save_fig_path)

print('Best loss: {}'.format(best_loss))
print('Best L2 dif: {}'.format(best_dif))

