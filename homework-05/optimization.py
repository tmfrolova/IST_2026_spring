import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize.linesearch import scalar_search_wolfe2
import time
from collections import defaultdict


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        def phi(alpha):
            return oracle.func_directional(x_k, d_k, alpha)

        def derphi(alpha):
            return oracle.grad_directional(x_k, d_k, alpha)

        if self._method == 'Constant':
            return self.c

        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            c1 = self.c1
            phi0 = phi(0.0)
            dphi0 = derphi(0.0)
            while phi(alpha) > phi0 + c1 * alpha * dphi0:
                alpha /= 2.0
            return alpha

        elif self._method == 'Wolfe':
            alpha0 = previous_alpha if previous_alpha is not None else self.alpha_0

            def phi_scaled(t):
                return phi(t * alpha0)

            def derphi_scaled(t):
                return derphi(t * alpha0) * alpha0

            result = scalar_search_wolfe2(
                phi_scaled, derphi_scaled, c1=self.c1, c2=self.c2
            )
            t = result[0]

            if t is None:
                alpha = alpha0
                phi0 = phi(0.0)
                dphi0 = derphi(0.0)
                while phi(alpha) > phi0 + self.c1 * alpha * dphi0:
                    alpha /= 2.0
                return alpha
            else:
                return t * alpha0
        else:
            raise ValueError('Unknown line search method')


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    grad_k = oracle.grad(x_0)
    grad_norm_sq = np.dot(grad_k, grad_k)
    grad_norm0_sq = grad_norm_sq if grad_norm_sq != 0 else 1.0

    start_time = time.time()
    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_norm_sq))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))

    if display:
        print("iter\tfunc\tgrad_norm")
        print(f"0\t{oracle.func(x_k):.6e}\t{np.sqrt(grad_norm_sq):.6e}")

    message = 'iterations_exceeded'
    for it in range(max_iter):
        # Проверка в начале итерации
        if grad_norm_sq <= tolerance * grad_norm0_sq:
            message = 'success'
            break

        d_k = -grad_k
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if alpha is None:
            message = 'computational_error'
            break

        x_new = x_k + alpha * d_k
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            message = 'computational_error'
            break
        x_k = x_new

        grad_k = oracle.grad(x_k)
        grad_norm_sq = np.dot(grad_k, grad_k)

        if trace:
            current_time = time.time() - start_time
            history['time'].append(current_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print(f"{it+1}\t{oracle.func(x_k):.6e}\t{np.sqrt(grad_norm_sq):.6e}")

        
        if grad_norm_sq <= tolerance * grad_norm0_sq:
            message = 'success'
            break

    return x_k, message, history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    grad_k = oracle.grad(x_0)
    grad_norm_sq = np.dot(grad_k, grad_k)
    grad_norm0_sq = grad_norm_sq if grad_norm_sq != 0 else 1.0

    start_time = time.time()
    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_norm_sq))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))

    if display:
        print("iter\tfunc\tgrad_norm")
        print(f"0\t{oracle.func(x_k):.6e}\t{np.sqrt(grad_norm_sq):.6e}")

    message = 'iterations_exceeded'
    for it in range(max_iter):
        # Проверка в начале итерации
        if grad_norm_sq <= tolerance * grad_norm0_sq:
            message = 'success'
            break

        try:
            H = oracle.hess(x_k)
            L, lower = scipy.linalg.cho_factor(H, lower=True)
            d_k = scipy.linalg.cho_solve((L, lower), -grad_k)
        except LinAlgError:
            message = 'computational_error'
            break

        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if alpha is None:
            message = 'computational_error'
            break

        x_new = x_k + alpha * d_k
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            message = 'computational_error'
            break
        x_k = x_new

        grad_k = oracle.grad(x_k)
        grad_norm_sq = np.dot(grad_k, grad_k)

        if trace:
            current_time = time.time() - start_time
            history['time'].append(current_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print(f"{it+1}\t{oracle.func(x_k):.6e}\t{np.sqrt(grad_norm_sq):.6e}")

        if grad_norm_sq <= tolerance * grad_norm0_sq:
            message = 'success'
            break

    return x_k, message, history
