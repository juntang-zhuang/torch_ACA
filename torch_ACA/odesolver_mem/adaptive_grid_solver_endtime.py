"""
This file contains a class of ODE solvers, which support "checkpoint" strategy to save memory.
However, denoting the initial time as t0 and end time as t1, this file only supports evaluate at t1.
t1 can be either greater or smaller than t0.
"""
import abc
import torch
import copy
import numpy as np
from torch.autograd import Variable
from torch import nn
from ..utils import monotonic
import copy
from ..misc import _dot_product, _scaled_dot_product, _interp_evaluate, _interp_fit

__all__ = ['RK12', 'RK23', "RK45", "Dopri5"]
# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

# The following code deals with the situation where the input and output of the derivative function is a tensor,
# currently not support tuple

class AdaptiveGridSolver(nn.Module):
    __metaclass__ = abc.ABCMeta
    order = NotImplemented

    def __init__(self, func, t0=0.0, t1=1.0, h=None, rtol=1e-3, atol=1e-6, neval_max=500000,
                 print_neval=False, print_direction=False, step_dif_ratio=1e-3, safety=None,
                 delete_graph = True, regenerate_graph = False, dense_output = False,
                 ):
        """
        ----------------
        :param func: callable
                the function to compute derivative, should be the form  derivative = func(t,y), y is a tensor
        :param t0: float
                initial time
        :param t1:float
                ending time
        :param h: float
                initial stepsize, could be none
        :param rtol: float
                relative error tolerance
        :param atol: float
                absolute error tolerance
        :param neval_max: int
                maximum number of evaluations, typically set as an extermely large number, e.g. 500,000
        :param print_neval: bool
                print number of evaluations or not
        :param print_direction: bool
                print direction of time (if t0 < t1, print 1; if t0 > t1, print -1)
        :param step_dif_ratio: float
                A ratio to avoid dead loop.
                if abs(old_step_size - new_step_size) < step_dif_ratio AND error > tolerance,
                then accept current stepsize and continue
        :param safety: float,
                same as scipy.odeint, used to adjut stepsize
        :param delete_graph, bool, whether delete redundant computation graph
        :param regenerate_graph, bool, whether re-generate computation graph using calculated grids
        ----------------
        """
        super(AdaptiveGridSolver, self).__init__()
        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.h = h
        self.rtol = rtol
        self.atol = atol
        self.neval_max = neval_max
        self.print_neval = print_neval
        self.neval = 0
        # if two stepsizes are too similar, not update it, otherwise stuck in a loop
        self.step_dif_ratio = step_dif_ratio
        self.delete_graph = delete_graph
        self.regenerate_graph = regenerate_graph

        # set time direction, forward-in-time is 1.0, reverse-time is -1.0
        if self.t1 > self.t0:
            self.time_direction = 1.0
            if print_direction:
                print('Forward-time integration')
        else:
            self.time_direction = -1.0
            if print_direction:
                print("Reverse-time integration")

        # same as the safety factor used in scipy.odeint
        if safety is not None:
            self.safety = safety
        else:
            self.safety = SAFETY

    def norm(self, x):
        """
        Calculate l2 norm per element
        """
        return torch.sqrt(torch.mean(x ** 2))

    def select_initial_step_scipy(self, t0, y0, f0):
        """Empirically select a good initial step.
        The algorithm is described in [1]_.
        Parameters
        ----------
        fun : callable
            Right-hand side of the system.
        t0 : float
            Initial value of the independent variable.
        y0 : ndarray, shape (n,)
            Initial value of the dependent variable.
        f0 : ndarray, shape (n,)
            Initial value of the derivative, i. e. ``fun(t0, y0)``.
        direction : float
            Integration direction.
        order : float
            Error estimator order. It means that the error controlled by the
            algorithm is proportional to ``step_size ** (order + 1)`.
        rtol : float
            Desired relative tolerance.
        atol : float
            Desired absolute tolerance.
        Returns
        -------
        h_abs : float
            Absolute value of the suggested initial step.
        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4.
        """
        if isinstance(y0, tuple) or isinstance(y0, list):
            print(
                'Current version only support y to be tensor-type, not list or tuple. Please concatenate into a single tensor')

        scale = self.atol + torch.abs(y0) * self.rtol
        d0 = self.norm(y0 / scale)
        d1 = self.norm(f0 / scale)
        if d0.item() < 1e-5 or d1.item() < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * self.time_direction * f0
        f1 = self.func(t0 + h0 * self.time_direction, y1)
        d2 = self.norm((f1 - f0) / scale) / h0

        if d1.item() <= 1e-15 and d2.item() <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1.item(), d2.item())) ** (1 / (self.order + 1))

        return min(100 * h0, h1)

    def delete_local_computation_graph(self, inputs):
        for i in inputs:
            i.set_()
            del i
        torch.cuda.empty_cache()
        return

    def adapt_stepsize(self, y, y_new, error, h_abs, step_accepted):
        """
        Adaptively modify the step size, code is modified from scipy.integrate package
        :param y:
        :param y_new:
        :param error:
        :param h_abs: step size
        :return: step_accepted: True if h_abs is acceptable. If False, set it as False, re-update h_abs
                 h_abs:  step size
        """
        scale = self.atol + torch.max(torch.abs(y), torch.abs(y_new)) * self.rtol
        error_norm = self.norm(error / scale).item()

        if error_norm == 0.0:
            factor = MAX_FACTOR
            step_accepted = True

        elif error_norm < 1:
            factor = min(MAX_FACTOR, max(1, self.safety * error_norm ** (-1 / (self.order + 1))))
            step_accepted = True

        else:
            factor = max(MIN_FACTOR, self.safety * error_norm ** (-1 / (self.order + 1)))
            step_accepted = False

        h_abs = h_abs * factor

        if torch.is_tensor(h_abs):
            h_abs = float(h_abs.item())

        return h_abs, step_accepted

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

    def integrate(self, y0, predefine_steps=None, return_steps=False, t_eval=None):
        ###############################################################################
        #         before integrate, initialize, choose initial stepsize               #
        ###############################################################################
        all_evaluations = []  # record outputs at t_eval

        self.before_integrate(y0, t_eval)

        is_stiff = False
        if isinstance(y0, tuple) or isinstance(y0, list):
            use_tuple = True
            print(
                'Currently only support tensor input, y cannot be tupe or list. Please concatenate into a single tensor.')

        state0 = self.func.state_dict()
        y_current = y0
        t_current = self.t0

        if self.h is not None:
            h_current = self.h
        else:
            # select initial step
            _y0 = Variable(y0.clone().detach(), requires_grad=False)

            with torch.no_grad():
                _f0 = self.func(self.t0, _y0)
                h_current = self.select_initial_step_scipy(t_current, _y0, _f0)

            self.delete_local_computation_graph([_y0, _f0])

        self.neval = 0  # number of evaluation steps
        self.func.load_state_dict(state0)

        #####################################################################################
        #             Step forward in time, if steps are not predefined                     #
        #####################################################################################
        if predefine_steps is None:  # time steps not specified, automatically modify stepsize

            steps = []
            step_accepted = False
            # advance a small step in time
            while abs(t_current-self.t0) <= abs(self.t1-self.t0) and abs(t_current + h_current * self.time_direction -self.t0) < abs(self.t1-self.t0) and self.neval < self.neval_max:
                # if not self.keep_small_step:
                step_accepted = False

                self.neval += 1
                h_new = h_current
                state0 = self.func.state_dict()
                n_try = 0

                #########################################################################
                #                   Determine optimal stepsize                          #
                #########################################################################
                while not step_accepted:
                    n_try += 1

                    if n_try >= self.neval_max:  # if is stiff, use predefined stepsize, not sure if this works well
                        is_stiff = True

                    if is_stiff:
                        h_new = min(self.h, abs(self.t1 - t_current))
                        step_accepted = True
                        print('Stiff problem, please use other solvers')

                    #####################################################################
                    #                   Delete redundant computation graph              #
                    #####################################################################

                    # detach y in order to avoid extra unused computation graphs
                    with torch.no_grad():

                        y_detach = Variable(y_current.clone().detach(), requires_grad=False)

                        h_current = h_new  # .clone().detach()

                        _y_new, _error, _variables = self.step(self.func, t_current, h_current * self.time_direction,
                                                               y_detach, return_variables=True)

                        h_new, step_accepted = self.adapt_stepsize(y_detach, _y_new, _error, h_current, step_accepted)

                        if not step_accepted:
                            if abs(h_new - h_current) / (h_current) < self.step_dif_ratio:
                                step_accepted = True

                        self.delete_local_computation_graph([y_detach, _y_new, _error] + list(_variables))

                    # restore state dict to before integrate
                    self.func.load_state_dict(state0)

                ##########################################################################
                #                         step forward                                   #
                ##########################################################################
                if abs(t_current + h_current * self.time_direction - self.t0) > abs(self.t1 - self.t0):
                    # if next step is beyond integration time, break
                    break
                else:
                    y_old = y_current
                    y_current, error, variables = self.step(self.func, t_current, h_current * self.time_direction,
                                                        y_current, return_variables=True)
                    #self.delete_local_computation_graph([ _error] + list(_variables))

                    t_current = t_current + h_current * self.time_direction
                    steps.append(t_current)

                # update stepsize
                h_current = h_new

            # if t_current < self.t1, make the last move
            if abs(t_current - self.t0) < abs(self.t1 - self.t0):
                step_current = self.t1 - t_current
                y_current, error, variables = self.step(self.func, t_current,step_current,
                                                        y_current, return_variables=True)
                #self.delete_local_computation_graph([_error] + list(_variables))

                t_current = self.t1
                steps.append(t_current)

            ##################################################################################
            #           If regenerate computation graph using estimated stepsizes            #
            ##################################################################################
            if self.regenerate_graph:
                y_current = self.integrate_predefined_grids(y0, predefine_steps=steps,
                                                                  return_steps=return_steps, t_eval=t_eval)
        ######################################################################################
        #                    Integrate when stepsizes are pre-defined                        #
        ######################################################################################
        else:  # time steps are pre-defined
            steps = predefine_steps
            y_current = self.integrate_predefined_grids(y0, predefine_steps=predefine_steps,
                                                              return_steps=return_steps, t_eval=t_eval)

        if self.print_neval:
            print('Number of evaluations: {} \n'.format(self.neval))

        if return_steps:
            return y_current, steps
        else:
            return y_current

    def integrate_predefined_grids(self, y0, predefine_steps=None, return_steps=False, t_eval=None):

        all_evaluations = []
        # print(len(predefine_steps))
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)

        self.before_integrate(y0, t_eval)

        time_points = predefine_steps

        # advance a small step in time
        t_current = self.t0
        y_current = y0
        # print(steps)
        for point in time_points:
            self.neval += 1
            y_old = y_current
            # print(y_current.shape)
            y_current, error, variables = self.step(self.func, t_current, (point - t_current), y_current,
                                                    return_variables=True)

            t_current = point

        return y_current

    def concate_results(self, outs):
        outs = torch.stack(outs, 0)
        return outs

    def update_t_end(self):
        # update t_end
        if self.t_eval is None or len(self.t_eval) == 0:
            self.t_end = None
        else:
            self.t_end = self.t_eval.pop(0)

    def step(self, func, t, dt, y, return_variables=False):
        pass

    def update_dense_state(self, t_old, t_new, t_eval, y, y_new, k):
        pass

    def interpolate(self, t_old, t_new, t_eval, y, y_new, k):
        # compute Q, correspond to _dense_output_impl in scipy
        # Q = self.K.T.dot(self.P)
        K = torch.stack(k, dim=1)  # Nx(n_stages+1)x...
        shape = y.shape
        K = K.view(shape[0], self.n_stages + 1, -1)  # Nx(n_stages +1)x-1
        K = K.permute(0, -1, 1)  # Nx -1 x (n_stages+1)

        # self.P.shape = (n_stages+1)xn_stages
        Q = torch.matmul(K, self.P.to(y.device))  # Nx-1xn_stages

        x = abs(t_eval - t_old) / abs(t_new - t_old)
        if np.array(t_eval).ndim == 0:
            p = np.tile(x, Q.shape[-1])
            p = np.cumprod(p)  # p.shape = n_stages

        p = torch.from_numpy(p).float().to(y.device)  # n_stages

        dif = float(t_new - t_old) * torch.matmul(Q, p)  # Nx-1
        dif = dif.view(y.shape)

        out = y + dif
        return out


class RK12(AdaptiveGridSolver):
    """
    Constants follow wikipedia
    """
    order = 1
    n_stages = 2
    C = np.array([1])
    A = [np.array([1])]
    B = np.array([1 / 2, 1 / 2])
    E = np.array([1 / 2, -1 / 2, 0])
    C_MID = [0.5, 0.0]

    def update_dense_state(self, t_old, t_new, t_eval, y0, y1, k):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = abs(t_new - t_old)  # .type_as(y0)
        y_mid = y0 + _scaled_dot_product(1.0, self.C_MID, k)
        f0 = k[0] / dt
        f1 = k[1] / dt
        coefficients = _interp_fit([y0], [y1], [y_mid], [f0], [f1], dt)
        self.coefficients = coefficients
        return

    def interpolate(self, t_old, t_new, t_eval, y0, y1, k):
        # super(Dopri5, self).interpolate(t_old, t_new, t_eval, y0, k)
        # recalculate y
        out = _interp_evaluate(coefficients=self.coefficients, t0=t_old, t1=t_new, t=t_eval)
        return out[0]

    def step(self, func, t, dt, y, return_variables=False):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt, y + 1.0 * k1)
        out1 = y + k1 * 0.5 + k2 * 0.5
        error = -0.5 * k1 + 0.5 * k2

        if return_variables:
            return out1, error, [k1, k2]
        else:
            return out1, error


class RK23(AdaptiveGridSolver):
    """
    Constants follow scipy implementation, https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Kutta's_third-order_method
    """
    order = 2
    n_stages = 3
    C = np.array([1 / 2, 3 / 4])
    A = [np.array([1 / 2]),
         np.array([0, 3 / 4])]
    B = np.array([2 / 9, 1 / 3, 4 / 9])
    E = np.array([5 / 72, -1 / 12, -1 / 9, 1 / 8])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2 / 3],
                  [0, 4 / 3, -8 / 9],
                  [0, -1, 1]])
    P = torch.from_numpy(P).float()

    def step(self, func, t, dt, y, return_variables=False):

        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        k3 = dt * func(t + dt * 0.75, y + 0.75 * k2)
        k4 = dt * func(t + dt, y + 2. / 9. * k1 + 1. / 3. * k2 + 4. / 9. * k3)
        out1 = y + 2. / 9. * k1 + 1. / 3. * k2 + 4. / 9. * k3
        error = 5/72 * k1 - 1/12*k2 -1/9 * k3 + 1/8 * k4

        if return_variables:
            return out1, error, [k1, k2, k3, k4]
        else:
            return out1, error


class RK45(AdaptiveGridSolver):
    """
    Constants follow wikipedia, https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Kutta's_third-order_method
    Fehlberg's method
    """
    order = 4
    n_stages = 6
    C = np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
    A = [np.array([1 / 5]),
         np.array([3 / 40, 9 / 40]),
         np.array([44 / 45, -56 / 15, 32 / 9]),
         np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
         np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656])]
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525,
                  1 / 40])
    # Corresponds to the optimum value of c_6 from [2]_.

    P = np.array([
        [1, -8048581381 / 2820520608, 8663915743 / 2820520608,
         -12715105075 / 11282082432],
        [0, 0, 0, 0],
        [0, 131558114200 / 32700410799, -68118460800 / 10900136933,
         87487479700 / 32700410799],
        [0, -1754552775 / 470086768, 14199869525 / 1410260304,
         -10690763975 / 1880347072],
        [0, 127303824393 / 49829197408, -318862633887 / 49829197408,
         701980252875 / 199316789632],
        [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]])

    P = torch.from_numpy(P).float()

    def step(self, func, t, dt, y, return_variables=False):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 5.0, y + 1.0 / 5.0 * k1)
        k3 = dt * func(t + dt * 3.0 / 10.0, y + 3.0 / 40.0 * k1 + 9.0 / 40.0 * k2)
        k4 = dt * func(t + dt * 4.0 / 5.0, y + 44.0 / 45.0 * k1 - 56.0 / 15. * k2 + 32. / 9. * k3)
        k5 = dt * func(t + dt * 8.0 / 9.0,
                       y + 19372. / 6561. * k1 - 25360.0 / 2187.0 * k2 + 64448. / 6561. * k3 - 212. / 729. * k4)
        k6 = dt * func(t + dt * 1.0,
                       y + 9017. / 3168.0 * k1 - 355.0 / 33.0 * k2 + 46732. / 5247. * k3 + 49. / 176. * k4 - 5103. / 18656. * k5)
        k7 = dt * func(t + dt * 1.0,
                       y + 35 / 384 * k1 + 0 * k2 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)

        out1 = y + 35. / 384. * k1 + 0. * k2 + 500. / 1113. * k3 + 125. / 192. * k4 - 2187. / 6784. * k5 + 11. / 84. * k6
        error = -71 / 57600 * k1 + 0 * k2 + 71 / 16695 * k3 - 71 / 1920 * k4 + 17253 / 339200 * k5 - 22 / 525 * k6 + 1 / 40 * k7

        if return_variables:
            return out1, error, [k1, k2, k3, k4, k5, k6, k7]
        else:
            return out1, error


class Dopri5(AdaptiveGridSolver):
    """
    Constants from https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/dopri5.py
    """
    order = 4
    n_stages = 7
    C = np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.])
    A = [np.array([1 / 5]),
         np.array([3 / 40, 9 / 40]),
         np.array([44 / 45, -56 / 15, 32 / 9]),
         np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
         np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
         np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])]
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])

    E = np.array([35 / 384 - 5179 / 57600, 0, 500 / 1113 - 7571 / 16695, 125 / 192 - 393 / 640,
                  -2187 / 6784 - -92097 / 339200, 11 / 84 - 187 / 2100, -1. / 40., 0])

    DPS_C_MID = [
        6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
        187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
    ]

    def update_dense_state(self, t_old, t_new, t_eval, y0, y1, k):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = abs(t_new - t_old)  # .type_as(y0)
        y_mid = y0 + _scaled_dot_product(1.0, self.DPS_C_MID, k)
        f0 = k[0] / dt
        f1 = k[1] / dt
        coefficients = _interp_fit([y0], [y1], [y_mid], [f0], [f1], dt)
        self.coefficients = coefficients
        return

    def interpolate(self, t_old, t_new, t_eval, y0, y1, k):
        # super(Dopri5, self).interpolate(t_old, t_new, t_eval, y0, k)
        # recalculate y
        out = _interp_evaluate(coefficients=self.coefficients, t0=t_old, t1=t_new, t=t_eval)
        return out[0]

    def step(self, func, t, dt, y, return_variables=False):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 5, y + 1 / 5 * k1)
        k3 = dt * func(t + dt * 3 / 10, y + 3 / 40 * k1 + 9.0 / 40.0 * k2)
        k4 = dt * func(t + dt * 4. / 5., y + 44. / 45. * k1 - 56. / 15. * k2 + 32. / 9. * k3)
        k5 = dt * func(t + dt * 8. / 9.,
                       y + 19372. / 6561. * k1 - 25360. / 2187. * k2 + 64448. / 6561. * k3 - 212. / 729. * k4)

        k6 = dt * func(t + dt,
                       y + 9017. / 3168. * k1 - 355. / 33. * k2 + 46732. / 5247. * k3 + 49. / 176. * k4 - 5103. / 18656. * k5)
        k7 = dt * func(t + dt,
                       y + 35. / 384. * k1 + 0 * k2 + 500. / 1113. * k3 + 125. / 192. * k4 - 2187. / 6784. * k5 + 11. / 84. * k6)

        out1 = y + 35. / 384. * k1 + 0 * k2 + 500. / 1113. * k3 + 125. / 192. * k4 - 2187. / 6784. * k5 + 11. / 84. * k6
        error = (35 / 384 - 5179 / 57600) * k1 + 0 * k2 + (500 / 1113 - 7571 / 16695) * k3 + (
                    125 / 192 - 393 / 640) * k4 + \
                (-2187 / 6784 + 92097 / 339200) * k5 + (11 / 84 - 187 / 2100) * k6 - 1 / 40 * k7

        if return_variables:
            return out1, error, [k1, k2, k3, k4, k5, k6, k7]
        else:
            return out1, error
