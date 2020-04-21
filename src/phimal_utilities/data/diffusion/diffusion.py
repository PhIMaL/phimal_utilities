from numpy import pi
import torch


def DiffusionGaussian(x, t, D, x0, sigma):
    u = (2 * pi * sigma**2 + 4 * pi * D * t)**(-1/2) * torch.exp(-(x - x0)**2/(2 * sigma**2 + 4 * D * t))
    return u
