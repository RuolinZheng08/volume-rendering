import numpy as np

# my modules
from utils import unlerp, lerp, quantize

class Ray():
    def start(self, idx_horizontal, idx_vertical, convolution, context):
        """
        rndRayStart
        """
        self.sample_idx = 0 # k-th sample along this ray
        self.step_view_len = None
        # len-4 vectors, with last entry either np.nan by default
        # or manually set to 1 for view-to-world matrix multiplication
        self.step_view = None
        self.pos_view_init = None
        self.pos_world_init = None

        self.result = np.full(4, context.outside_val) # 4 for RGBA
        self.result_cache = None
        self.result_curr = None

        # for blend mode over
        self.transparency = None

        camera = context.camera
        ray_img = np.empty((4, 1))
        ray_img[0] = (camera.img_plane_width / 2) * \
        lerp(-1, 1, -0.5, idx_horizontal, camera.img_plane_size[0] - 0.5)
        ray_img[1] = (camera.img_plane_height / 2) * \
        lerp(1, -1, -0.5, idx_vertical, camera.img_plane_size[1] - 0.5)
        ray_img[2] = -camera.dist
        ray_img[3] = np.nan

        if camera.ortho:
            self.pos_view_init = np.array([[ray_img[0, 0], ray_img[0, 1],
            -camera.near_clip_view, np.nan]]).T
            self.step_view = np.array([[0, 0, -context.plane_sep, np.nan]]).T
        else: # perspective
            scale = camera.near_clip_view / camera.dist
            self.pos_view_init = scale * ray_img
            scale = context.plane_sep / camera.dist
            self.step_view = scale * ray_img
        self.step_view_len = np.linalg.norm(self.step_view)

        # manually set last entry to 1
        self.pos_view_init[3] = 1
        # convert view-space initial position to world-space
        self.pos_world_init = camera.VtoW @ self.pos_view_init

    def go(self, idx_horizontal, idx_vertical, convolution, context):
        self.start(idx_horizontal, idx_vertical, convolution, context)
        keepgoing = True
        while keepgoing:
            keepgoing = self.step(convolution, context)
        return self.result

    def step(self, convolution, context):
        """
        rndProbeRgbaLit
        returns a boolean, keepgoing
        stores result in self.result
        """
        keepgoing = True
        camera = context.camera
        pos_view = self.pos_view_init + self.sample_idx * self.step_view
        # stop when -p_n > fcv
        if -pos_view[2] > camera.far_clip_view:
            return False # no need to keep going
        self.sample_idx += 1

        # manually set last entry to 1 for matrix multiplication
        pos_view[3] = 1
        pos_world = camera.VtoW @ pos_view
        convolution.evaluate(pos_world[0, 0], pos_world[1, 0], pos_world[2, 0], context)
        if not convolution.inside:
            return keepgoing # skip this sample, proceed to the next

        convo_grad_len = np.linalg.norm(convolution.gradient)
        # quantize the convo result to get the LUT lookup index
        transfer_func = context.transfer_func
        lut_idx = quantize(transfer_func.vmin, convolution.value,
        transfer_func.vmax, transfer_func.len)

        rgba = transfer_func.rgba[:, lut_idx]

        # clamp opacity to between 0 and 1
        clamped = np.clip(rgba[3], 0, 1)
        # opacity correction
        # corrected = 1 - ((1 - clamped) ^ (delta / unit_step))
        corrected = 1 - pow((1 - clamped), self.step_view_len / transfer_func.unit_step)
        rgba[3] = np.clip(corrected, 0, 1)

        # no need to do Blinn-Phong if corrected opacity is 0
        # since color won't change
        if corrected == 0:
            rgb_lit = rgba
        else:
            if camera.ortho: # viewer direction is context.camera.n
                viewer_dir = camera.n
            else: # perspective
                pos_world_dir = self.pos_world_init - pos_world
                pos_world_dir /= np.linalg(pos_world_dir) # normalize
                viewer_dir = pos_world_dir
            rgb_lit = self.blinn_phong(rgba[:3], convolution.gradient,
            viewer_dir, context)

        # depth cueing
        gamma = lerp(0, 1, camera.near_clip_view, -pos_view[0, 2], camera.far_clip_view)
        dcn = context.params_light.depth_color_near
        dcf = context.params_light.depth_color_far
        color_lerped = lerp(dcn, dcf, gamma)
        # multiply component-wise 3-vec of rgb and copy opacity over
        self.result_curr = np.append(rgb_lit * color_lerped, rgba[3])

        keepgoing = self.blend()
        return keepgoing

    def blend(self):
        """
        rndBlendOver
        returns a boolean, keepgoing
        """
        keepgoing = True
        transparency_curr = 1 - self.result_curr[3] # 1 - opacity
        # check if this is the first step, where self.result_cache is None
        if self.result_cache is None:
            self.result = self.result_curr
            self.transparency = transparency_curr
        else:
            rgb_premultiplied = self.result_cache[3] * self.result_cache[:3]
            rgb_curr = self.transparency * self.result_curr[3] * self.result_curr[:3]
            rgb_composite = rgb_premultiplied + rgb_curr
            self.transparency *= transparency_curr
            opacity = 1 - self.transparency
            if opacity == 0: # avoid division by zero producing NaNs
                self.result = np.zeros(4)
            else:
                self.result[:3] = 1 / opacity * rgb_composite
                self.result[3] = opacity
            # stop early if opacity > alpha_near_one
            if opacity > context.transfer_func.alpha_near_one:
                self.transparency = 0
                keepgoing = False
        self.result_cache = self.result
        return keepgoing # True by default unless set otherwise

    def blinn_phong(self, rgb_in, gradient, viewer_dir, context):
        """
        returns a 3-vector of rgb values
        """
        # note that rgb_in stores the color of the material, c_M
        light = context.light
        params_light = context.params_light
        rgb_out = params_light.k_ambient * rgb_in
        if light.num == 0 or \
        (params_light.k_ambient == 0 and params_light.k_specular == 0):
            return rgb_out # done
        gradient_len = np.linalg.norm(gradient)
        if gradient_len == 0:
            return rgb_out # done

        # surface normal N = -g/|g|
        normal = -gradient / gradient_len
        for i in range(light.num):
            light_col = light.rgb[i]
            light_dir = light.xyz[i]
            # compute diffuse
            # component-wise c_M * c_L
            prod = rgb_in * light.rgb[i]
            scale = max(0, np.dot(normal, light_dir))
            diffuse = params_light.k_diffuse * scale * prod

            # compute specular
            # halfway between V and L
            halfway = (viewer_dir + light_dir)
            halfway /= np.linalg.norm(halfway) # normalize
            scale = pow(max(0, np.dot(normal, halfway), params_light.p_shininess))
            specular = params_light.k_specular * scale * light_col

            rgb_out += diffuse + specular

        return rgb_out
