import numpy as np
from phimal_utilities.data.burgers.burgers import BurgersDelta
from phimal_utilities.data.base import Dataset


x = np.linspace(-1, 1, 50)
t = np.linspace(1e-4, 5, 20)

x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
v = 0.1
A = 1.0

dataset = Dataset(BurgersDelta, v=v, A=A)


dataset.library(x_grid.reshape(-1, 1), t.reshape(-1, 1))
