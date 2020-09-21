from torch_ACA.odesolver import odesolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy

np.random.seed(0)

alpha = 0.5
beta  = 0.5
delta = 0.5
gamma = 0.5

def lotka_volterra(t, x):
    if torch.is_tensor(x):
        d = torch.zeros_like(x)
    else:
        d = np.zeros(2)
    d[0] = alpha * x[0] - beta * x[0] * x[1]
    d[1] = delta * x[0] * x[1] - gamma * x[1]
    return d

x0 = (np.random.random() + np.ones(2)).astype(np.float32)

t = np.linspace(0, 10., 1000)

x = solve_ivp(lotka_volterra, (0.0, 10.0), x0, method='RK45', t_eval=t, rtol=1e-3, atol=1e-6).y

plt.plot(x[0], x[1])

x0 = torch.from_numpy(x0)

options = {}
options.update({'method': 'RK45'})
options.update({'t0': 0.0})
options.update({'t1': 10.0})
options.update({'rtol': 1e-3})
options.update({'atol': 1e-6})
options.update({'t_eval':t.tolist()})

x = odesolve(lotka_volterra, x0, options).data.cpu().numpy()

plt.plot(x[:, 0], x[:, 1],'-.')
plt.show()