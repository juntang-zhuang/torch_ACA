import abc
import torch
import copy
import numpy as np
from torch import nn

__all__ = ['Euler','RK2','RK4']

class FixedGridSolver(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, t0=0.0, t1=1.0, h = 0.1, rtol=1e-3, atol=1e-6, neval_max=500000,
                 print_neval=False, print_direction=False, step_dif_ratio=1e-3, safety=None,delete_graph = True,
                 regenerate_graph = True):
        super(FixedGridSolver, self).__init__()
        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.h = h
        if h is None:
            print('Stepsize h is required for fixed grid solvers')
        self.Nt = int(abs(t1 - t0)/h)
        self.print_neval = print_neval
        self.neval = 0

        # set time direction, forward-in-time is 1.0, reverse-time is -1.0
        if self.t1 > self.t0:
            self.time_direction = 1.0
        else:
            self.time_direction = -1.0

    @abc.abstractmethod
    def step(self, func, t, dt, y):
        pass

    def integrate(self, y0, predefine_steps = None, return_steps=False):
        if isinstance(y0, tuple) or isinstance(y0, list):
            use_tuple = True
        else:
            use_tuple = False

        y_current = y0
        dt = self.h
        self.neval = 0

        if predefine_steps is None: # use steps defined by h
            steps = []
            t0 = self.t0
            # advance a small step in time
            for n in range(self.Nt):
                self.neval += 1
                y_current = self.step(self.func, t0, self.h*self.time_direction, y_current)
                t0 = t0 + self.h*self.time_direction
                steps.append(self.h)
        else: # use specified step size
            steps = predefine_steps
            # advance a small step in time
            t0 = self.t0
            y_current = y0
            for step in steps:
                self.neval += 1
                y_current = self.step(self.func, t0, step*self.time_direction, y_current)
                t0 = t0 + step * self.time_direction

        if self.print_neval:
            print('Number of evaluations: {}\n'.format(self.neval))

        if return_steps:
            return y_current, steps
        else:
            return y_current


class Euler(FixedGridSolver):
    order = 1
    def step(self, func, t, dt, y):
        out = func(t,y)
        out = y + dt * out
        return out

class RK2(FixedGridSolver):
    order = 2
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        out = y + k2
        return out

class RK4(FixedGridSolver):
    order = 4
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        k3 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k2)
        k4 = dt * func(t + dt, y + k3)
        out = y + 1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 + 1.0 / 6.0 * k4
        return out

