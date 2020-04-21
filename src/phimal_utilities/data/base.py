import numpy as np


class Dataset:
    def __init__(self, solution, **kwargs):
        self.solution = solution  # set solution
        self.parameters = kwargs  # set solution parameters

    def solution(self, x, t):
        '''Generation solution.'''
        return self.data(output=0, coordinate=0, order=0)(x, t, **self.parameters)

    def data(self, output, coordinate, order):
        '''Simple wrapper function which makes it easier to grab the right function
        from the solution instance.'''
        return getattr(self.solution, f'u_{output}_{coordinate}_{order}')

    def library(self, x, t):
        ''' Returns library with 3rd order derivs and 2nd order polynomial'''
        u = self.data(output=0, coordinate=0, order=0)(x, t, **self.parameters)
        u_x = self.data(output=0, coordinate=1, order=1)(x, t, **self.parameters)
        u_xx = self.data(output=0, coordinate=1, order=2)(x, t, **self.parameters)
        u_xxx = self.data(output=0, coordinate=1, order=3)(x, t, **self.parameters)

        derivs = np.concatenate([np.ones_like(u), u_x, u_xx, u_xxx], axis=1)
        theta = np.concatenate([derivs, u * derivs, u**2 * derivs], axis=1)

        return theta

    def time_deriv(self, x, t):
        ''' Return time derivative'''
        return self.data(output=0, coordinate=0, order=1)(x, t, **self.parameters)
