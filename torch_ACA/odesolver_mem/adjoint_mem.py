
import torch
import torch.nn as nn
from .ode_solver_endtime import odesolve_endtime
from torch.autograd import Variable
import copy
__all__ = ['odesolve_adjoint']

def flatten_params(params):
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

def flatten_params_grad(params, params_ref):
    _params = [p for p in params]
    _params_ref = [p for p in params_ref]
    flat_params = [p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(_params, _params_ref)]

    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

class Checkpointing_Adjoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        z0, func, flat_params, options= args[:-3], args[-3], args[-2], args[-1]

        if isinstance(z0,tuple):
            if len(z0) == 1:
                z0 = z0[0]

        ctx.func = func
        state0 = func.state_dict()
        ctx.state0 = state0
        if isinstance(z0, tuple):
            ctx.z0 = tuple([_z0.data for _z0 in z0])
        else:
            ctx.z0 = z0.data

        ctx.options = options

        with torch.no_grad():
            solver = odesolve_endtime(func, z0, options, return_solver=True, regenerate_graph = False)
            #solver.func.load_state_dict(state0)
            ans, steps = solver.integrate(z0, return_steps=True)

        ctx.steps = steps
        #ctx.ans = ans

        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        if isinstance(ctx.z0, tuple):
            z0 = tuple([Variable(_z0, requires_grad=True) for _z0 in ctx.z0])
        else:
            z0 = Variable(ctx.z0, requires_grad=True)

        options = ctx.options
        func = ctx.func
        f_params = func.parameters()

        steps, state0 = ctx.steps, ctx.state0

        func.load_state_dict(state0)

        if isinstance(z0, tuple) or isinstance(z0, list):
            use_tuple = True
        else:
            use_tuple = False

        z = z0

        solver = odesolve_endtime(func, z, options, return_solver=True)

        # record inputs to each step
        inputs = []
        inputs.append(z)

        #t0 = solver.t0
        t_current = solver.t0
        y_current = z
        for point in steps:
            solver.neval += 1
            # print(y_current.shape)
            with torch.no_grad():
                y_current, error, variables = solver.step(solver.func, t_current, point - t_current, y_current, return_variables=True)
                t_current = point

            if not use_tuple:
                inputs.append(Variable(y_current.data, requires_grad = True))
            else:
                inputs.append([Variable(_y.data, requires_grad=True) for _y in y_current])

            if use_tuple:
                solver.delete_local_computation_graph(list(error) + list(variables))
            else:
                solver.delete_local_computation_graph([error] + list(variables))

        # delete the gradient directly applied to the original input
        # if use tuple, input is directly concatenated with output
        grad_output = list(grad_output)
        if use_tuple:
            input_direct_grad = grad_output[0][0,...]
            grad_output[0] = grad_output[0][1,...]
        grad_output = tuple(grad_output)

        ###################################
        #print(steps)
        # note that steps does not include the start point, need to include it
        steps = [options['t0']] + steps
        # now two list corresponds, steps = [t0, teval1, teval2, ... tevaln, t1]
        #                           inputs = [z0, z1, z2, ... , z_out]
        ###################################

        inputs.pop(-1)
        steps2 = copy.deepcopy(steps)
        steps2.pop(0)
        steps.pop(-1)

        # steps = [t0, eval1, eval2, ... evaln, t1], after pop is [t0, eval1, ... evaln]
        # steps2 = [t0, eval1, eval2, ... evaln, t1], after pop is [eval1, ... evaln, t1]

        # after reverse, they are
        # steps = [evaln, evaln-1, ... eval2, eval1, t0]
        # steps2 = [t1, evaln, ... eval2, eval1s]

        param_grads = []
        inputs.reverse()
        steps.reverse()
        steps2.reverse()

        assert len(inputs) == len(steps) == len(steps2), print('len inputs {}, len steps {}, len steps2 {}'.format(len(inputs), len(steps), len(steps2)))

        for input, point, point2 in zip(inputs, steps, steps2):
            if not use_tuple:
                input = Variable(input, requires_grad = True)
            else:
                input = [Variable(_, requires_grad = True) for _ in input]
                input = tuple(input)

            with torch.enable_grad():
                #print(type(z))
                y, error, variables = solver.step(solver.func, point, point2 - point, input, return_variables=True)

                param_grad = torch.autograd.grad(
                    y, f_params,
                    grad_output, retain_graph=True)

                grad_output = torch.autograd.grad(
                 y,  input,
                 grad_output)

                param_grads.append(param_grad)

                if use_tuple:
                    solver.delete_local_computation_graph(list(y) + list(error) + list(variables))
                else:
                    solver.delete_local_computation_graph([y, error] + list(variables))

        # sum up gradients w.r.t parameters at each step, stored in out2
        out2 = param_grads[0]
        for i in range(1, len(param_grads)):
            for _1, _2 in zip([*out2], [*param_grads[i]]):
                _1 += _2

        # attach direct gradient w.r.t input
        if use_tuple:
            grad_output = list(grad_output)
            # add grad output to direct gradient
            if input_direct_grad is not None:
                grad_output[0] = input_direct_grad + grad_output[0]#torch.stack((input_direct_grad, grad_output[0]), dim=0)

            grad_output = tuple(grad_output)
        out = tuple([*grad_output] + [None, flatten_params_grad(out2, func.parameters()), None])

        return out
        #return  out1[0], out1[1], None, flatten_params_grad(out2, func.parameters()), None


def odesolve_adjoint(func, z0, options = None):

    flat_params = flatten_params(func.parameters())
    if isinstance(z0, tuple) or isinstance(z0, list):
        zs = Checkpointing_Adjoint.apply(*z0, func, flat_params, options)
    else:
        zs = Checkpointing_Adjoint.apply(z0, func, flat_params, options)
    return zs
