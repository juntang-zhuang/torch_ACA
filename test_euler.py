from torch_ACA.odesolver import odesolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy

H = 50
ts = 0.1 * torch.arange(H+1,dtype=torch.float32)

def odef(t,x):
    return x

s0 = torch.zeros(10,1)

options = {}
options.update({'method': 'Euler'})
options.update({'h': 0.01})
options.update({'t0': ts.tolist()[0] })
options.update({'t1': ts.tolist()[-1] })
options.update({'t_eval':ts.tolist()})
st = odesolve(odef, s0, options) # s0 is [N,n] & st is [49,N,n]