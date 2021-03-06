class Camera():
    def __init__(self, fr, at, up, near_clip, far_clip, field_of_view, img_plane_size, aspect_ratio, ortho):
        """
        field_of_view: degrees
        """
        self.fr = fr
        self.at = at
        self.up = up
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.field_of_view = field_of_view
        self.img_plane_size = img_plane_size # 2-tuple
        self.aspect_ratio = aspect_ratio
        self.ortho = ortho

        # do the computations like in rndCameraUpdate
        fr_minus_at = fr - at
        # d = |fr - at|
        self.dist = np.linalg.norm(fr_minus_at)
        # n = (fr - at) / |fr - at|
        self.n = fr_minus_at / self.dist
        # u = up x n / |up x n|
        up_cross_n = np.cross(up, self.n)
        self.u = up_cross_n / np.linalg.norm(up_cross_n)
        # v = n x u
        self.v = np.cross(self.n, self.u)
        # assemble view-to-world matrix
        mat = np.hstack([
            self.u, self.v, self.n, self.fr
        ])
        self.VtoW = np.append(mat, [[0, 0, 0, 1]], axis=0)
        # ncv, fcv: near_clip_view, far_clip_view
        self.near_clip_view = self.near_clip + self.dist
        self.far_clip_view = self.far_clip + self.dist
        # height and width of the image plane, different from sizes
        # hght = 2d tan(FOV / 2), FOV must be in radians
        radians = np.radians(self.field_of_view)
        self.img_plane_height = 2 * self.d * np.tan(radians / 2)
        # wdth = ar hght
        self.img_plane_width = self.aspect_ratio * self.img_plane_height
