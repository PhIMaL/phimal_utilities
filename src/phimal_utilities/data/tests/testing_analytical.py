# Script used to test if the solution is implemented correctly.
# The idea is that least squares on the analytical library gives the
# correct result up to machine precision.

import numpy as np
from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersSawtooth, BurgersDelta, BurgersCos


def test_solution(dataset, v, x=None, t=None):
    if x is None:
        x = np.linspace(-1, 1, 50)
    if t is None:
        t = np.linspace(1e-4, 1, 20)
    x_grid, t_grid = np.meshgrid(x, t)

    theta = dataset.library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))
    dt = dataset.time_deriv(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))
    coeffs = np.linalg.lstsq(theta, dt, rcond=None)[0]

    true_coeffs = np.zeros((12, 1))
    true_coeffs[2] = v
    true_coeffs[5] = -1.0

    return np.allclose(true_coeffs, coeffs), coeffs

v = 0.1
#dataset = Dataset(BurgersDelta, v=v, A=1.0)
#dataset = Dataset(BurgersCos, v=v, a=0.1, b=0.1, k=2)
#dataset = Dataset(BurgersSawtooth, v=v)

x_saw = np.linspace(0, 2*np.pi, 50) # saw tooth only valid on specific domain
t_saw = np.linspace(0.0, 0.5, 20)

# Actual test
assert test_solution(dataset, v=v)[0] is True, 'Calculated coefficients not correct.'
print('Test succesfully run')
