import numpy as np
from scipy.special import erfc


class BurgersDelta():
    ''' Class to generate analytical solutions of Burgers equation with delta peak initial condition. Names should be
    u_[output]_[coordinate]_[order], with time being coordinate 0. I.e. the function which calculates the second derivative of the third output w.r.t to time
    is called u_2_0_2. Non derivatives are called with u_[output]_0_0.

    Good source: https://www.iist.ac.in/sites/default/files/people/IN08026/Burgers_equation_viscous.pdf'''

    @staticmethod
    def u_0_0_0(x, t, v, A):
        '''Calculates solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)

        solution = np.sqrt(v/(np.pi*t)) * ((np.exp(R) - 1) * np.exp(-z**2)) / (1 + (np.exp(R) - 1)/2*erfc(z))
        return solution

    @staticmethod
    def u_0_1_1(x, t, v, A):
        '''Calculates first order spatial derivative of solution.'''
        z = x/np.sqrt(4*v*t)

        u = BurgersDelta.u_0_0_0(x, t, v, A)
        u_x = 1/np.sqrt(4*v*t) * (np.sqrt(t/v)*u**2-2*z*u)
        return u_x

    @staticmethod
    def u_0_1_2(x, t, v, A):
        '''Calculates second order spatial derivative of solution.'''
        z = x/np.sqrt(4*v*t)

        u = BurgersDelta.u_0_0_0(x, t, v, A)
        u_x = BurgersDelta.u_0_1_1(x, t, v, A)
        u_xx = 1/np.sqrt(4*v*t) * (-2*u/np.sqrt(4*v*t) - 2*z*u_x + 2*np.sqrt(t/v)*u*u_x) # could be written shorter, but then get NaNs due to inversions

        return u_xx

    @staticmethod
    def u_0_1_3(x, t, v, A):
        '''Calculates third order spatial derivative of solution.'''
        z = x/np.sqrt(4*v*t)

        u = BurgersDelta.u_0_0_0(x, t, v, A)
        u_x = BurgersDelta.u_0_1_1(x, t, v, A)
        u_xx = BurgersDelta.u_0_1_2(x, t, v, A)
        u_xxx = 1/np.sqrt(4*v*t) * (-4/np.sqrt(4*v*t) * u_x + 2 * np.sqrt(t/v)*u_x**2 + u_xx*(-2*z+2*np.sqrt(t/v)*u)) # could be written shorter, but then get NaNs due to inversions

        return u_xxx

    @staticmethod
    def u_0_0_1(x, t, v, A):
        '''Calculates first order temporal derivative of solution.'''
        u = BurgersDelta.u_0_0_0(x, t, v, A)
        u_x = BurgersDelta.u_0_1_1(x, t, v, A)
        u_xx = BurgersDelta.u_0_1_2(x, t, v, A)
        u_t = v * u_xx - u * u_x
        return u_t
