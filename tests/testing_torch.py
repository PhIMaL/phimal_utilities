import numpy as np
from phimal_utilities.data import Dataset
from phimal_utilities.data.diffusion import DiffusionGaussian
from phimal_utilities.data.burgers import BurgersDelta, BurgersCos, BurgersSawtooth
from phimal_utilities.data.kdv import KdVSoliton

x = np.linspace(-5, 5, 1000)
t = np.linspace(0.0, 2.0, 100)

x_grid, t_grid = np.meshgrid(x, t)

dataset = Dataset(BurgersDelta, v=0.1, A=1.0)
dataset = Dataset(BurgersCos, v=0.1, a=0.1, b=0.1, k=2)
dataset = Dataset(BurgersSawtooth, v=0.1)
#dataset = Dataset(KdVSoliton, c=5.0, a = -1.0, b=1)

dataset.generate_solution(x_grid, t_grid).shape
dataset.parameters

dataset.time_deriv(x_grid, t_grid).shape

theta = dataset.library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), poly_order=2, deriv_order=2)
dt = dataset.time_deriv(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))


theta.shape
np.linalg.lstsq(theta, dt, rcond=None)[0]




X_train, y_train = dataset.create_dataset(x_grid, t_grid, n_samples=0, noise=0.05)

y_train.shape

from phimal_utilities.analysis import load_tensorboard
