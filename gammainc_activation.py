# implemention of gammainc as activation function in pytorch

import torch
import torchquad


class Regularized_lower_incomplete_gamma(torch.autograd.Function):
    @staticmethod
    def forward(a, x):
        output = torch.special.gammainc(a, x)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, x = inputs
        ctx.save_for_backward(a, x)

    @staticmethod
    def backward(ctx, grad_out):
        '''
        compute the gradient of gammainc(a, x) w.r.t. a and x
        the gradient of gammainc(a, x) w.r.t a is
        1 / gamma(a) * int_0^x exp(-t) * t ^(a-1) * ln(t) dt - diff(gamma(a)) / gamma(a)^2 * int_0^x exp(-t) * t^(a-1) dt
        the gradient of gammainc(a, x) w.r.t. x is
        1 / gamma(a) * exp(-x) * x^(a-1)
        '''

        def grad_wrt_a(a, x):
            '''
            use torchquad to compute the integral value
            '''

            def integrand_1(a, t):
                '''
                define the integrand of exp(-t) * t^(a-1)
                '''
                return torch.exp(-1 * t) * torch.pow(t, a - 1)

            def integrand_2(a, t):
                '''
                define the integrand of exp(-t) * t^(a-1) * ln(t)
                '''
                return torch.exp(-1 * t) * torch.pow(t, a - 1) * torch.log(t)

            boole = torchquad.Boole()
            device = x.device
            # set tol to 1e-12 to avoid the integral start from 0 since the integral value is inf at 0
            tol = torch.tensor([1e-12]).to(device)
            # use gammaln(a) to obtain the factor of the integral and divide the integral domain into 10001 points
            gamma_function_value = torch.exp(torch.special.gammaln(a))
            integral_1 = torch.stack([boole.integrate(lambda t: integrand_1(a, t.to(device)), dim=1, N=10001,
                                                      integration_domain=torch.stack([tol, x_bound], dim=-1),
                                                      backend="torch"
                                                      ) for x_bound in x])
            integral_1 = -1 * torch.multiply(gamma_function_value, integral_1)

            integral_2 = torch.stack([boole.integrate(lambda t: integrand_2(a, t.to(device)), dim=1, N=10001,
                                                      integration_domain=torch.stack([tol, x_bound], dim=-1),
                                                      backend="torch"
                                                      ) for x_bound in x])
            return torch.multiply(torch.div(1, gamma_function_value), integral_1 + integral_2)

        def grad_wrt_x(a, x):
            '''
            compute the value of the gradient of gammainc(a, x) w.r.t. x
            '''
            gamma_function_value = torch.exp(torch.special.gammaln(a))
            factor_1 = torch.div(1, gamma_function_value)
            factor_2 = torch.exp(-1 * x) * torch.pow(x, a - 1)
            return factor_1 * factor_2

        a, x = ctx.saved_tensors
        grad_a = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_a = grad_out * grad_wrt_a(a, x).t()
        if ctx.needs_input_grad[1]:
            grad_x = grad_out * grad_wrt_x(a, x)

        return grad_a, grad_x


class Gammainac_activation(torch.nn.Module):
    '''
    Wrap the autograd of gammainc
    '''
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.Tensor(1))
        self.sigmoid_activation = torch.nn.Sigmoid()
        self.scale_factor = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.a, val=1.0)
        torch.nn.init.constant_(self.scale_factor, val=4.5)

    def forward(self, x):
        '''
        use sigmoid function to rescale the input domain of x
        '''
        x = torch.multiply(self.scale_factor, self.sigmoid_activation(x))
        return Regularized_lower_incomplete_gamma.apply(self.a, x)
