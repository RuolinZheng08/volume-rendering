import numpy as np

class Convolution():
    def __init__(self):
        """
        needgrad always true b/c rndProbeRgbaLit
        """
        self.inside = True # set to False if convo cannot be evaluated
        self.value = None
        self.gradient = None # world-space gradients

    def evaluate(x_world, y_world, z_world, context):
        size_x, size_y, size_z = context.volume.data.shape
        pos_world = np.array([x_world, y_world, z_world, 1]]).T # column vec
        self.pos_index = context.WtoI @ pos_world
        x_index, y_index, z_index = self.pos_index.squeeze().tolist()
        # cache intermediate computation
        idx_start = context.idx_start
        idx_end = context.idx_end
        cache_length = idx_end - idx_start + 1

        kern_cache_x = np.empty(cache_length)
        kern_cache_y = np.empty(cache_length)
        kern_deriv_cache_x = np.empty(cache_length)
        kern_deriv_cache_y = np.empty(cache_length)

        kernel = context.kernel

        # compute convolution
        if kernel.support & 1: # odd support
            xn = np.floor(x_index + 0.5)
            yn = np.floor(y_index + 0.5)
            zn = np.floor(z_index + 0.5)
        else:
            xn = np.floor(x_index)
            yn = np.floor(y_index)
            zn = np.floor(z_index)
        xalpha = x_index - xn
        yalpha = y_index - yn
        zalpha = z_index - zn

        convo_result = 0 # assign to self.value if convo is valid
        # column vector to accumulate gradient during convo
        gradient_index = np.zeros((1, 3))

        # the main convo loop
        # index into context.volume[vol_idx_x, vol_idx_y, vol_idx_z]
        # outermost loop, slow axis
        for zi in range(idx_start, idx_end + 1):
            vol_idx_z = zn + zi
            if not self.inside or vol_idx_z >= size_z:
                self.inside = False
                break
            # compute convo result for current z
            kern_res_z = kernel.evaluate(zalpha - zi)
            kern_deriv_res_z = kernel.evaluate_derivative(zalpha - zi)

            # inner loop, faster axis
            for yi in range(idx_start, idx_end + 1):
                vol_idx_y = yn + yi
                if not self.inside or vol_idx_y >= size_y:
                    self.inside = False
                    break
                if zi == idx_start: # first iter, compute cache
                    kern_cache_y[yi - idx_start] = kernel.evaluate(yalpha - yi)
                    kern_deriv_cache_y[yi - idx_start] = \
                    kernel.evaluate_derivative(yalpha - yi)
                kern_res_y = kern_cache_y[yi - idx_start]
                kern_deriv_res_y = kern_derive_cache_y[yi - idx_start]

                # innermost loop, fastest axis
                for xi in range(idx_start, idx_end + 1):
                    vol_idx_x = xn + xi
                    if not self.inside or vol_idx_x >= size_x:
                        self.inside = False
                        break
                    # first iter, compute cache
                    if zi == idx_start and yi == idx_start:
                        kern_cache_x[xi - idx_start] = kernel.evaluate(xalpha - xi)
                        kern_deriv_cache_x[xi - idx_start] = \
                        kernel.evaluate_derivative(xalpha - xi)
                    kern_res_x = kern_cache_x[xi - idx_start]
                    kern_deriv_res_x = kern_deriv_cache_x[xi - idx_start]

                    val = context.volume[vol_idx_x, vol_idx_y, vol_idx_z]
                    # accumulate convo result
                    convo_result += val * kern_res_x * kern_res_y * kern_res_z
                    # accumulate gradients
                    gradient_index[0, 0] += val * \
                    kern_deriv_res_x * kern_res_y * kern_res_z
                    gradient_index[0, 1] += val * \
                    kern_res_x * kern_deriv_res_y * kern_res_z
                    gradient_index[0, 2] += val * \
                    kern_res_x * kern_res_y * kern_deriv_res_z
                    # end innermost loop
                # end inner loop
            # end outer loop
        if self.inside: # convo is valid
            self.value = convo_result
            self.gradient = context.gradient_ItoW @ gradient_index
        # no return value
