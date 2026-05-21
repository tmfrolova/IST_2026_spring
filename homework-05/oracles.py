import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import issparse, diags, eye


class BaseSmoothOracle(object):
    def func(self, x):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        z = self.matvec_Ax(x)
        t = -self.b * z
        loss = np.mean(np.logaddexp(0, t))
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        z = self.matvec_Ax(x)
        p = expit(-self.b * z)
        s = -self.b * p
        grad = self.matvec_ATx(s) / self.m + self.regcoef * x
        return grad

    def hess(self, x):
        z = self.matvec_Ax(x)
        p = expit(-self.b * z)
        w = p * (1 - p)
        H = self.matmat_ATsA(w)  # может быть разреженной или плотной
        # приводим к плотному массиву, чтобы удобно добавить регуляризацию
        if issparse(H):
            H = H.toarray()
        H = H / self.m
        n = x.shape[0]
        H += self.regcoef * np.eye(n)
        return H


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    
    pass


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    def matmat_ATsA(s):
        # Вычисляет A.T @ diag(s) @ A
        if issparse(A):
            # Для разреженных матриц используем dot с диагональной
            return A.T @ (diags(s) @ A)
        else:
            return A.T @ (s[:, None] * A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    n = x.shape[0]
    grad = np.zeros(n)
    f0 = func(x)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f1 = func(x + e_i)
        grad[i] = (f1 - f0) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    n = x.shape[0]
    hess = np.zeros((n, n))
    f0 = func(x)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_i = func(x + e_i)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = eps
            f_j = func(x + e_j)
            f_ij = func(x + e_i + e_j)
            hess[i, j] = (f_ij - f_i - f_j + f0) / (eps * eps)
    return hess
