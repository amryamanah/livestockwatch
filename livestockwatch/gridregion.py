from collections import namedtuple

import numpy as np
import cv2

from .config import IMG_WIDTH, IMG_HEIGHT

Point = namedtuple("Point", ["x", "y"])
Region = namedtuple("Region", ["A", "B", "C", "D"])


class GridRegion:
    def __init__(self, partition_number=3):
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.w_step = self.width / partition_number
        self.h_step = self.height / partition_number
        self.x_slice = [int(x) for x in np.arange(0, self.width + 1, self.w_step)]
        self.y_slice = [int(x) for x in np.arange(0, self.height + 1, self.h_step)]

        self.lst_region = (
            Region(Point(0, 0), Point(426, 0), Point(426, 320), Point(0, 320)),
            Region(Point(426, 0), Point(853, 0), Point(853, 320), Point(420, 320)),
            Region(Point(853, 0), Point(1280, 0), Point(1280, 320), Point(853, 320)),

            Region(Point(0, 320), Point(426, 320), Point(426, 640), Point(0, 640)),
            Region(Point(426, 320), Point(853, 320), Point(853, 640), Point(426, 640)),
            Region(Point(853, 320), Point(1280, 320), Point(1280, 640), Point(853, 640)),

            Region(Point(0, 640), Point(426, 640), Point(426, 960), Point(0, 960)),
            Region(Point(426, 640), Point(853, 640), Point(853, 960), Point(426, 960)),
            Region(Point(853, 640), Point(1280, 640), Point(1280, 960), Point(853, 960)),
        )

    def draw_region(self, image):
        for num, region in enumerate(self.lst_region):
            cv2.rectangle(image, (region.A.x, region.A.y), (region.C.x, region.C.y), (255, 0, 0), 3)
            center_x = int(region.A.x + (self.w_step / 2))
            center_y = int(region.A.y + (self.h_step / 2))
            cv2.putText(image, str(num + 1).upper(), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 4,
                        (128, 255, 0), 5, cv2.LINE_AA)
        return image

    def get_centroid_region(self, centroid, image=None):
        region_id = np.nan
        center_x = centroid[0]
        center_y = centroid[1]

        for num, region in enumerate(self.lst_region):
            if region.A.x <= center_x <= region.C.x:
                if region.A.y <= center_y <= region.C.y:
                    region_id = int(num+1)
                    break

        cv2.putText(image, str(region_id).upper(), (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 4,
                    (0, 0, 255), 5, cv2.LINE_AA)

        return region_id, image

