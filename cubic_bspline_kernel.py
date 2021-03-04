# kernel.c

def cubic_bspline_eval(xx):
    ret = 0
    x = fabs(xx)
    if x < 1:
        ret = 2 / 3 + x * x * (-1 + x / 2)
    elif x < 2:
        x -= 1
        ret = 1 / 6 + x * (-1 / 2 + x * (1 / 2 - x / 6))
    return ret

def cubic_bspline_deriv_eval(xx):
    """
    derivative of cubic B-spline kernel
    """
    ret = 0
    x = abs(xx)
    if x < 1:
        ret = x * (-2 + x * (3 / 2))
    elif x < 2:
        x -= 1
        ret = -1 / 2 + x * (1 - x / 2)
    return (xx < 0) ? -ret : ret
