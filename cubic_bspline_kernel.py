# kernel.c
import numpy as np

class CubicBsplineKernel():
    def __init__(self):
        self.support = 4

    def evaluate(self, xx):
        ret = 0
        x = np.fabs(xx)
        if x < 1:
            ret = 2 / 3 + x * x * (-1 + x / 2)
        elif x < 2:
            x -= 1
            ret = 1 / 6 + x * (-1 / 2 + x * (1 / 2 - x / 6))
        return ret

    def apply(self, xs):
        """
        vectorized version of evaluate, xs is a vector
        """
        ret = np.zeros(xs.shape)
        x = np.fabs(xs)
        ret = np.where(x < 1, 2 / 3 + x * x * (-1 + x / 2), ret)
        x_minus_one = x - 1
        ret = np.where((x >= 1) & (x < 2),
        1 / 6 + x_minus_one * (-1 / 2 + x_minus_one * (1 / 2 - x_minus_one / 6)), ret)
        return ret

    def evaluate_derivative(self, xx):
        """
        derivative of cubic B-spline kernel
        """
        ret = 0
        x = np.fabs(xx)
        if x < 1:
            ret = x * (-2 + x * (3 / 2))
        elif x < 2:
            x -= 1
            ret = -1 / 2 + x * (1 - x / 2)
        if xx < 0:
            return -ret
        else:
            return ret

    def apply_derivative(self, xs):
        ret = np.zeros(xs.shape)
        x = np.fabs(xs)
        ret = np.where(x < 1, x * (-2 + x * (3 / 2)), ret)
        x_minus_one = x - 1
        ret = np.where((x >= 1) & (x < 2),
        -1 / 2 + x_minus_one * (1 - x_minus_one / 2), ret)
        ret = np.where(xs < 0, -ret, ret)
        return ret
