from IPython import embed


class ImageFormation:

    def __init__(self):
        # effective image sensor dimension
        self.isdim_x = 4.8
        self.isdim_y = 3.6

        # half x FOV = 21.8 in degree
        self.fov_x = 43.6

        # half y FOV = 16.7 in degree
        self.fov_y = 33.4

        # f_length in mm
        self.f_length = 6

        # image dimension
        self.imdim_x = 1280
        self.imdim_y = 960

        # degree per pixel
        self.dpx_x, self.dpx_y = self._calculate_deg_per_px()

    def _calculate_deg_per_px(self):
        deg_per_px_x = self.fov_x / self.imdim_x
        deg_per_px_y = self.fov_y / self.imdim_y
        return deg_per_px_x, deg_per_px_y





