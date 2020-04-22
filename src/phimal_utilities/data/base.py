import torch
from numpy import ndarray


def pytorch_func(function):
    '''Decorator to automatically transform arrays to tensors and back'''
    def wrapper(self, *args, **kwargs):
        torch_args = [torch.tensor(arg, requires_grad=True, dtype=torch.float64) for arg in args if type(arg) is ndarray]
        torch_kwargs = {key: torch.tensor(kwarg, requires_grad=True, dtype=torch.float64) for key, kwarg in kwargs.items() if type(kwarg) is ndarray}
        result = function(self, *torch_args, **torch_kwargs)
        return result.detach().numpy()
    return wrapper


class Dataset:
    def __init__(self, solution, **kwargs):
        self.solution = solution  # set solution
        self.parameters = kwargs  # set solution parameters

    @pytorch_func
    def generate_solution(self, x, t):
        '''Generation solution.'''
        u = self.solution(x, t, **self.parameters)
        return u

    @pytorch_func
    def time_deriv(self, x, t):
        u = self.solution(x, t, **self.parameters)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        return u_t

    @pytorch_func
    def library(self, x, t, poly_order=2, deriv_order=2):
        ''' Returns library with 3rd order derivs and 2nd order polynomial'''
        assert ((x.shape[1] == 1) & (t.shape[1] == 1)), 'x and t should have shape (n_samples x 1)'

        u = self.solution(x, t, **self.parameters)

        # Polynomial part
        poly_library = torch.ones_like(u)
        for order in torch.arange(1, poly_order+1):
            poly_library = torch.cat((poly_library, poly_library[:, order-1:order] * u), dim=1)

        # derivative part
        if deriv_order == 0:
            deriv_library = torch.ones_like(u)
        else:
            du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            deriv_library = torch.cat((torch.ones_like(u), du), dim=1)
            if deriv_order > 1:
                for order in torch.arange(1, deriv_order):
                    du = torch.autograd.grad(deriv_library[:, order:order+1], x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                    deriv_library = torch.cat((deriv_library, du), dim=1)

        # Making library
        theta = torch.matmul(poly_library[:, :, None], deriv_library[:, None, :]).reshape(u.shape[0], -1)
        return theta
