import numpy as np


class BurgersDelta:
    ''' Class to generate analytical solutions of Burgers equation with delta peak initial condition.
    This is the solution instance, so it only contains the logic required to calculate the solution and the library.

    Good source: https://www.iist.ac.in/sites/default/files/people/IN08026/Burgers_equation_viscous.pdf
    Note theres an error in the derivation, the term in front of the erfc
    needs to be sqrt(pi)/2, not sqrt(pi/2)''''

    @staticmethod
    def u(x, t, v, A):
        '''Calculates solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)

        solution = np.sqrt(v/(np.pi*t)) * ((np.exp(R) - 1) * np.exp(-z**2)) / (1 + (np.exp(R) - 1)/2*erfc(z))
        return solution

    @staticmethod
    def u_x(x, t, v, A):
        '''Calculates first order spatial derivative of solution.'''
        z = x/np.sqrt(4*v*t)

        u = BurgersDelta.u(x, t, v, A)
        u_x = 1/np.sqrt(4*v*t) * (np.sqrt(t/v)*u**2-2*z*u)
        return u_x

    @staticmethod
    def u_xx(x, t, v, A):
        '''Calculates second order spatial derivative of solution.'''
        z = x/np.sqrt(4*v*t)

        u = BurgersDelta.u(x, t, v, A)
        u_x = BurgersDelta.u_x(x, t, v, A)
        u_xx = 1/np.sqrt(4*v*t) * (-2*u/np.sqrt(4*v*t) - 2*z*u_x + 2*np.sqrt(t/v)*u*u_x) # could be written shorter, but then get NaNs due to inversions
        return u_xx

    @staticmethod
    def u_xxx(x, t, v, A):
        '''Calculates third order spatial derivative of solution.'''
        z = x/np.sqrt(4*v*t)

        u = BurgersDelta.u(x, t, v, A)
        u_x = BurgersDelta.u_x(x, t, v, A)
        u_xx = BurgersDelta.u_xx(x, t, v, A)
        u_xxx = 1/np.sqrt(4*v*t) * (-4/np.sqrt(4*v*t) * u_x + 2 *np.sqrt(t/v)*u_x**2 + u_xx*(-2*z+2*np.sqrt(t/v)*u)) # could be written shorter, but then get NaNs due to inversions
        return u_xxx

    @staticmethod
    def u_t(x, t, v, A):
        '''Calculates first order temporal derivative of solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)

        u = BurgersDelta.u(x, t, v, A)
        u_x = BurgersDelta.u_x(x, t, v, A)
        u_xx = BurgersDelta.u_xx(x, t, v, A)
        u_t = v * u_xx - u * u_x
        return u_t
