import numpy as np
from phimal_utilities.data.burgers import BurgersDelta
from phimal_utilities.data.diffusion import DiffusionGaussian
from phimal_utilities.data import Dataset


x = np.linspace(-1, 1, 50)
t = np.linspace(1e-4, 5, 20)

x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
v = 0.1
A = 1.0

dataset = Dataset(DiffusionGaussian, D=1.0, sigma=0.1, x0=-0.1)


dataset.library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))
