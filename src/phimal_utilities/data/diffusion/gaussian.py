import numpy as np


class DiffusionGaussian():
    @staticmethod
    def u_0_0_0(x, t, D, x0, sigma):
        u = (2 * np.pi * sigma**2 + 4 * np.pi * D * t)**(-1/2) * np.exp(-(x - x0)**2/(2*sigma**2+4 * D * t))
        return u

    @staticmethod
    def u_0_1_1(x, t, D, x0, sigma):
        u = DiffusionGaussian.u_0_0_0(x, t, D, x0, sigma)

        u_x = -2 * (x - x0) / (2 * sigma**2 + 4 * D * t) * u
        return u_x

    @staticmethod
    def u_0_1_2(x, t, D, x0, sigma):
        u = DiffusionGaussian.u_0_0_0(x, t, D, x0, sigma)

        u_xx = ((x-x0)**2 - sigma**2 - 2 * D * t) / (sigma**2 + 2 * D * t)**2 * u
        return u_xx

    @staticmethod
    def u_0_1_3(x, t, D, x0, sigma):
        u = DiffusionGaussian.u_0_0_0(x, t, D, x0, sigma)

        u_xxx = -(x - x0) * (-3 * sigma**2 - 6 * D * t + (x - x0)**2) / (sigma**2 + 2 * D * t)**3 * u
        return u_xxx

    @staticmethod
    def u_0_0_1(x, t, D, x0, sigma):
        u_xx = u_0_1_2(x, t, D, x0, sigma)
        u_t = D * u_xx

        return u_t
