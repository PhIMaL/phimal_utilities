import torch
import numpy as np
from phimal_utilities.data import DatasetTorch
from phimal_utilities.data.diffusion import DiffusionGaussian

x0 = 0.0
sigma = 0.1
D = 1.0

x = np.linspace(-1, 1, 50)
t = np.linspace(1e-4, 5, 20)

x_grid, t_grid = np.meshgrid(x, t)

dataset = DatasetTorch(DiffusionGaussian, D=D, x0=x0, sigma=sigma)
dataset.generate_solution(x_grid, t_grid)
dataset.parameters

dataset.time_deriv(x_grid, t_grid).shape

theta = dataset.library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))
dt = dataset.time_deriv(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))
np.linalg.lstsq(
