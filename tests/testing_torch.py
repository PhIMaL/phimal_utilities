import numpy as np
from phimal_utilities.data import Dataset
from phimal_utilities.data.diffusion import DiffusionGaussian
from phimal_utilities.data.burgers import BurgersDelta, BurgersCos, BurgersSawtooth, BurgersDiscontinuous


x = np.linspace(0, 2*np.pi, 50)
t = np.linspace(1e-4, 0.5, 20)

x_grid, t_grid = np.meshgrid(x, t)

dataset = Dataset(BurgersDelta, v=0.1, A=1.0)
dataset = Dataset(BurgersCos, v=0.1, a=0.1, b=0.1, k=2)
dataset = Dataset(BurgersSawtooth, v=0.1)
dataset = Dataset(BurgersDiscontinuous, v=1.0, a=0.5, b=0.1)


dataset.generate_solution(x_grid, t_grid).shape
dataset.parameters

dataset.time_deriv(x_grid, t_grid).shape

theta = dataset.library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))
dt = dataset.time_deriv(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))


np.linalg.lstsq(theta, dt, rcond=None)[0]




x = np.linspace(0, 2*np.pi, 50)
t = np.linspace(1e-4, 0.5, 20)
dataset = Dataset(BurgersCos, v=0.1, a=0.1, b=0.1, k=2)
y_train, X_train = dataset.generate_deepmod_dataset(x, t, noise=0.05, n_samples=1000, random=True)


from phimal_utilities.tensorboard import load_tensorboard
df = load_tensorboard(path)
