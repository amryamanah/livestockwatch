import cv2
from skimage import exposure
from skimage import feature

from IPython import embed


class HogDescriptor:
    def __init__(self, orientation, pixels_per_cell, cells_per_block=1, normalise=True, visualise=False):
        self.orientation = orientation
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.normalise = normalise
        self.visualise = visualise

    def __str__(self):
        return "HOG"

    def describe(self, img):
        if self.visualise:
            (H, hogImage) = feature.hog(img, orientations=self.orientation,
                                        pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                                        cells_per_block=(self.cells_per_block, self.cells_per_block),
                                        normalise=self.normalise, visualise=self.visualise)
            hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
            hogImage = hogImage.astype("uint8")
            cv2.imshow("HOG Image", hogImage)
            cv2.waitKey(0)

            return H
        else:
            H = feature.hog(img, orientations=self.orientation,
                            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                            cells_per_block=(self.cells_per_block, self.cells_per_block),
                            normalise=self.normalise, visualise=self.visualise)
            return H