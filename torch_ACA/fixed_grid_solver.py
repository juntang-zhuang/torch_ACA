import abc
import torch
import copy
import numpy as np
from torch import nn
from .utils import monotonic

__all__ = ['Euler','RK2','RK4']

class FixedGridSolver(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, t0=0.0, t1=1.0, h = 0.1, rtol=1e-3, atol=1e-6, neval_max=500000,
                 print_neval=False, print_direction=False, step_dif_ratio=1e-3, safety=None,delete_graph = True,
                 regenerate_graph = True, dense_output = False):
        super(FixedGridSolver, self).__init__()
        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.h = h
        if h is None:
            print('Stepsize h is required for fixed grid solvers')
        self.Nt = round(abs(t1 - t0)/h)
        self.print_neval = print_neval
        self.neval = 0

        self.dense_output = dense_output

        if self.dense_output:
            if not hasattr(self, 'dense_states'):
                self.init_dense_states()

        # set time direction, forward-in-time is 1.0, reverse-time is -1.0
        if self.t1 > self.t0:
            self.time_direction = 1.0
        else:
            self.time_direction = -1.0

    def before_integrate(self, y0, t_eval):
        t_eval = copy.deepcopy(t_eval)

        if (t_eval is not None) and (len(t_eval) > 0):
            self.t_eval = t_eval
            assert isinstance(t_eval, list), 't_eval must be of type list'
            assert (self.t1 - t_eval[-1]) * (t_eval[0] - self.t0) >= 0, \
                'value of t_eval must be within t0<= t_eval <= t1'
            if len(t_eval) > 1:
                assert monotonic(t_eval), 't_eval muist be monotonic'
                assert (t_eval[-1] - t_eval[
                    0]) * self.time_direction > 0, 't_eval must be arranged in the same direction as [t0, t1]'
            self.t_end = t_eval[0]
            self.t_eval.pop(0)
        else:
            self.t_end = self.t1
            self.t_eval = None

    def update_t_end(self):
        # update t_end
        if self.t_eval is None or len(self.t_eval) == 0:
            self.t_end = None
        else:
            self.t_end = self.t_eval.pop(0)


    def delete_dense_states(self):
        if len(self.dense_states) > 0:
            if len(self.dense_states['t_start']) > 0:
                self.dense_states['t_start'].clear()
            if len(self.dense_states['t_end']) > 0:
                self.dense_states['t_end'].clear()
            if len(self.dense_states['y_start']) > 0:
                self.delete_local_computation_graph(self.dense_states['y_start'])
            if len(self.dense_states['y_end']) > 0:
                self.delete_local_computation_graph(self.dense_states['y_end'])
            if len(self.dense_states['variables']) > 0:
                for _variable in self.dense_states['variables']:
                    self.delete_local_computation_graph(list(_variable))
            if len(self.dense_states['coefficients']) > 0:
                for _coeff in self.dense_states['coefficients']:
                    self.delete_local_computation_graph(list(_coeff))

        self.init_dense_states()

    def init_dense_states(self):
        self.dense_states = {
            't_start': [],
            't_end':[],
            'y_start':[],
            'y_end':[],
            'variables':[],
            'coefficients':[],
        }

    def interpolate(self, t_old, t_new, t_eval, y0, y1, k):
        # linear interpolation
        out = (t_eval - t_old) * (y1 - y0) / (t_new - t_old) + y0
        return out

    @abc.abstractmethod
    def step(self, func, t, dt, y, return_variables = False):
        pass

    def update_dense_state(self, t_old, t_new, t_eval, y, y_new, k=None, save_current_step = True):
        if self.dense_output and save_current_step:
            self.dense_states['t_start'].append(copy.deepcopy(t_old))
            self.dense_states['t_end'].append(copy.deepcopy(t_new))
            self.dense_states['y_start'].append(y)
            self.dense_states['y_end'].append(y_new)
            self.dense_states['variables'].append(k)

    def integrate(self, y0, predefine_steps = None, return_steps=False, t_eval = None):
        self.before_integrate(y0, t_eval)

        # check if f is a function that returns tensor
        if isinstance(y0, tuple) or isinstance(y0, list):
            use_tuple = True
            print('Currently only support tensor functions, please concatenate all tensors into a single tensor in f')

        # determine integration steps
        if predefine_steps is None: # use steps defined by h
            steps = [self.t0 + (n+1)* abs(self.h) * self.time_direction for n in range(self.Nt)]
        else:
            steps = predefine_steps

        # integration in time
        all_evaluations = self.integrate_predefined_grids( y0, predefine_steps=steps, return_steps=False, t_eval = t_eval)
        out = self.concate_results(all_evaluations)

        if self.print_neval:
            print('Number of evaluations: {}\n'.format(self.neval))

        if return_steps:
            return out, steps
        else:
            return out

    def concate_results(self, outs):
        # concatenate into a tensor
        if len(outs) == 1:
            out = outs[0]
        elif len(outs) > 1:
            out = torch.stack(outs, 0)
        else:
            out = None
            print('Length of evaluated results is 0 in fixed-grid integration mode, please check')
        return out

    def integrate_predefined_grids(self, y0, predefine_steps=None, return_steps=False, t_eval=None):

        all_evaluations = []

        time_points = predefine_steps

        # advance a small step in time
        t_current = self.t0
        y_current = y0
        # print(steps)
        for point in time_points:
            self.neval += 1
            y_old = y_current
            # print(y_current.shape)
            y_current, variables = self.step(self.func, t_current, (point - t_current), y_current,return_variables=True)

            if self.dense_output:
                self.update_dense_state(t_current, point, self.t_end, y_old, y_current, variables)

            while (self.t_end is not None) and abs(point - self.t0) >= abs(self.t_end - self.t0) and \
                    abs(t_current - self.t0) <= abs(self.t_end - self.t0):  # if next step is beyond integration time
                # interpolate and record output
                all_evaluations.append(
                    self.interpolate(t_current, point, self.t_end, y_old, y_current, variables)
                )
                self.update_t_end()
            t_current = point

        # if have points outside the integration range
        while self.t_end is not None:
            print('Evaluation points outside integration range. Please re-specify t0 and t1 s.t. t0 < t_eval < t1 or t1 < t_eval < t0 STRICTLY, and use a FINER grid.')
            if not self.dense_output:
                print('DenseOutput mode is not enabled. ')
            else:
                print('Extrapolate in dense mode')
                all_evaluations.append(self.evaluate_dense_mode([self.t_end]))
            self.update_t_end()

        return all_evaluations

    def evaluate_dense_mode(self, t_eval):# evaluate at time points in t_eval, with dense mode.
        all_evaluations = []

        for _t_eval in t_eval:

            # find the correct interval for t
            ind = 0
            ind_found = False
            while ind < len(self.dense_states['t_start']):
                t_start, t_end = self.dense_states['t_start'][ind], self.dense_states['t_end'][ind],
                if abs(t_end - self.t0) >= abs(_t_eval - self.t0) and \
                    abs(t_start - self.t0) <= abs(_t_eval - self.t0):
                    ind_found = True
                    break
                else:
                    ind += 1

            if not ind_found:
                print('Evaluation time outside integration range.')
                if abs(self.dense_states['t_start'][0] - _t_eval) > abs(self.dense_states['t_start'][-1] - _t_eval):
                    ind = -1
                    print('Extrapolate using the last interval')
                else:
                    ind = 0
                    print('Extrapolate using the first interval')

            y_start, y_end, variables = self.dense_states['y_start'][ind], self.dense_states['y_end'][ind], \
                                        self.dense_states['variables'][ind]
            # evaluate by interpolation
            all_evaluations.append(
                self.interpolate(t_start, t_end, _t_eval, y_start, y_end, variables)
            )

        return self.concate_results(all_evaluations)

class Euler(FixedGridSolver):
    order = 1
    def step(self, func, t, dt, y, return_variables=False):
        out = func(t,y)
        out = y + dt * out
        if return_variables:
            return out, None
        else:
            return out

class RK2(FixedGridSolver):
    order = 2
    def step(self, func, t, dt, y, return_variables=False):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        out = y + k2
        if return_variables:
            return out, [k1, k2]
        else:
            return out

class RK4(FixedGridSolver):
    order = 4
    def step(self, func, t, dt, y, return_variables=False):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        k3 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k2)
        k4 = dt * func(t + dt, y + k3)
        out = y + 1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 + 1.0 / 6.0 * k4
        if return_variables:
            return out, [k1, k2, k3, k4]
        else:
            return out

