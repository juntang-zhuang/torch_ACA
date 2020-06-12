
from ..fixed_grid_solver import *
from .adaptive_grid_solver_endtime import *
from ..utils import extract_keys
__all__ = ['odesolve_endtime']

def odesolve_endtime(func, z0, options, return_solver=False, **kwargs):
    hyperparams = extract_keys(options)

    if options['method'] == 'Euler':
        solver = Euler(func,**hyperparams,**kwargs)
    elif options['method'] == 'RK2':
        solver = RK2(func,**hyperparams, **kwargs)
    elif options['method'] == 'RK4':
        solver = RK4(func,**hyperparams, **kwargs)
    elif options['method'] == 'RK12':
        solver = RK12(func,**hyperparams, **kwargs)
    elif options['method'] == 'RK23':
        solver = RK23(func, **hyperparams, **kwargs)
    elif options['method'] == 'RK45':
        solver = RK45(func, **hyperparams, **kwargs)
    elif options['method'] == 'Dopri5':
        solver = Dopri5(func, **hyperparams, **kwargs)
    else:
        print('Name of solver not found.')

    if return_solver: # return solver
        return solver
    else: # return integrated value
        z1 = solver.integrate(z0)#, t_eval = options['t_eval'])
        return z1
