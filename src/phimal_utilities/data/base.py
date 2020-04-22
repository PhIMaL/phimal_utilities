import torch


def pytorch_func(function):
    '''Decorator to automatically transform arrays to tensors and back'''
    def wrapper(self, *args, **kwargs):
        torch_args = [torch.tensor(arg, requires_grad=True, dtype=torch.float64) for arg in args]
        torch_kwargs = {key: torch.tensor(kwarg, requires_grad=True, dtype=torch.float64) for key, kwarg in kwargs.items()}
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
    def library(self, x, t):
        ''' Returns library with 3rd order derivs and 2nd order polynomial'''
        assert ((x.shape[1] == 1) & (t.shape[1] == 1)), 'x and t should have shape (n_samples x 1)'

        u = self.solution(x, t, **self.parameters)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx))[0]

        derivs = torch.cat([torch.ones_like(u), u_x, u_xx, u_xxx], dim=1)
        theta = torch.cat([derivs, u * derivs, u**2 * derivs], dim=1)

        return theta
