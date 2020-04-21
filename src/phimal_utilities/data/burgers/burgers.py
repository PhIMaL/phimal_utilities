import torch
from numpy import pi


def BurgersDelta(x, t, v, A):
    ''' Function to generate analytical solutions of Burgers equation with delta peak initial condition.

    Good source: https://www.iist.ac.in/sites/default/files/people/IN08026/Burgers_equation_viscous.pdf
    Note that this source has an error in the erfc prefactor, should be sqrt(pi)/2, not sqrt(pi/2).'''
    R = torch.tensor(A / (2 * v)) # otherwise throws error
    z = x / torch.sqrt(4 * v * t)

    u = torch.sqrt(v / (pi * t)) * ((torch.exp(R) - 1) * torch.exp(-z**2)) / (1 + (torch.exp(R) - 1) / 2 * torch.erfc(z))
    return u
