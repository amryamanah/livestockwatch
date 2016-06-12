import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
from .utils import equ_hist_color_image
from IPython import embed


class PupilAnalysisException(Exception):
    """Base class for Pupil Analysis"""
    pass


class NoPupilDetected(PupilAnalysisException):
    """Raised when no pupil is detected"""
    pass


class ImpartialPupilDetected(PupilAnalysisException):
    """Raised when impartial pupil is detected"""
    pass


class UnsupportedAnalysisMode(PupilAnalysisException):
    """Raised when impartial pupil is detected"""
    pass


def pupil_fit_ellipse(image, blur_kernel=5, close_kernel=7, open_kernel=22, debug=False):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    minor_axis = None
    major_axis = None
    cnt = None
    angle = None
    centroid = None
    ellipse = None

    image_untouched = image.copy()
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    blur = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    frame_to_thresh = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(frame_to_thresh, (97.5, 0, 0), (127.5, 255, 255))

    #
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closing)
    #
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening)

    (_, cnts, _) = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0 and cnts is not None:
        try:
            cnt = max(cnts, key=cv2.contourArea)
        except Exception as e:
            embed()

        area = cv2.contourArea(cnt)
        print(area)
        if area > 3000:
            image_untouched = equ_hist_color_image(image_untouched)
            cv2.drawContours(image_untouched, cnt, -1, (0, 0, 255), 3)

            ellipse = cv2.fitEllipse(cnt)
            centroid, (minor_axis, major_axis), angle = ellipse
            cv2.ellipse(image_untouched, (centroid, (minor_axis, major_axis), angle), (0, 255, 0), 2)
            if debug:
                cv2.imshow("Original", np.hstack([image_untouched, blur]))
                cv2.imshow("Result", np.hstack([thresh, closing, opening, mask]))

                cv2.waitKey(0)

    return centroid, major_axis, minor_axis, angle, image_untouched, cnt


def calculate_axis_skimage(img):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    bw = np.logical_or(img, mask)
    # plt.imshow(bw)
    # plt.show()

    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    props = None
    pupil_area = 0

    s = regionprops(label_image)
    if s:
        for region in s:

            if props is None:
                props = region
            else:
                if props.area < region.area:
                    props = region

        boundary = [0 for _ in range(4)]
        boundary[0], boundary[1], boundary[2], boundary[3] = props.bbox

        for x in boundary:
            if x <= 1:
                props = None

        if props:
            pupil_area = (props.major_axis_length / 2) * (props.minor_axis_length / 2) * np.pi
            # print(props.bbox)

        if not props.area and props.area < 5000:
            pupil_area = 0
            props = None
            print("Area = {} is less than 5000 px".format(props.area))
        if pupil_area == 0:
            props = None
    else:
        props = None

    return props, pupil_area


class Parameter:
    def __init__(self, hue_max, hue_min, sat_max=1.0, sat_min=0.0, val_max=1.0, val_min=0.0):
        self.hue_max = hue_max
        self.hue_min = hue_min
        self.sat_max = sat_max
        self.sat_min = sat_min
        self.val_max = val_max
        self.val_min = val_min


def create_mask(RGB, parameter):
    # Convert RGB image to chosen color space
    hsv_img = color.rgb2hsv(RGB)

    hue_min = parameter.hue_min
    hue_max = parameter.hue_max
    sat_min = parameter.sat_min
    sat_max = parameter.sat_max
    val_min = parameter.val_min
    val_max = parameter.val_max

    h = np.logical_and(
            (hsv_img[:, :, 0] >= hue_min),
            (hsv_img[:, :, 0] <= hue_max))
    s = np.logical_and(
            (hsv_img[:, :, 1] >= sat_min),
            (hsv_img[:, :, 1] <= sat_max))
    v = np.logical_and(
            (hsv_img[:, :, 2] >= val_min),
            (hsv_img[:, :, 2] <= val_max))

    BW = np.logical_and(h, np.logical_and(s, v))

    return BW


def detect_pupil(rgb, parameter):
    bw = create_mask(rgb, parameter)

    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    props = None
    pupil_area = 0

    s = regionprops(label_image)
    if s:
        for region in s:

            if props is None:
                props = region
            else:
                if props.area < region.area:
                    props = region

        boundary = [0 for _ in range(4)]
        boundary[0], boundary[1], boundary[2], boundary[3] = props.bbox

        for x in boundary:
            if x <= 1:
                props = None

        if props:
            pupil_area = (props.major_axis_length / 2) * (props.minor_axis_length / 2) * np.pi
            # print(props.bbox)

        if not props.area and props.area < 5000:
            pupil_area = 0
            props = None
            print("Area = {} is less than 5000 px".format(props.area))
        if pupil_area == 0:
            props = None
    else:
        props = None

    return props, pupil_area


def save_pupil_analysis_result(rgb, props, result_path):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_title("Area=%s" % props.area)
    ax.imshow(rgb)

    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)
    # print("attempt to save analysis result to : {}".format(result_path))
    fig.savefig(result_path, bbox_inches='tight')
    plt.close(fig)


class EllipseAnalysis(object):
    def __init__(self, major_axis, minor_axis):
        self.semi_major_axis = major_axis / 2.0
        self.semi_minor_axis = minor_axis / 2.0
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.eccentricity = self.calculate_eccentricity()
        self.ipr = self.calculate_ipr()

    def calculate_area(self):
        if self.semi_major_axis == 0:
            return 0
        else:
            area = np.pi * self.semi_major_axis * self.semi_minor_axis
            return area

    def calculate_perimeter(self):
        if self.semi_major_axis == 0:
            return 0
        else:
            perimeter = (2.00 * np.pi) * math.sqrt(((self.semi_major_axis ** 2) + (self.semi_minor_axis ** 2)) / 2.00)
            return perimeter

    def calculate_eccentricity(self):
        if self.semi_major_axis == 0:
            return 0
        else:
            eccentricity = math.sqrt((self.semi_major_axis ** 2) - (self.semi_minor_axis ** 2)) / self.semi_major_axis
            return eccentricity

    def calculate_ipr(self):
        if self.perimeter == 0:
            return 0
        else:
            ipr = (4 * np.pi * self.area) / (self.perimeter ** 2)
            return ipr


def ellipse_normalized_area(area, max_area):
    if max_area == 0:
        return 0
    else:
        return area / max_area


def ellipse_calculate_ca(area, max_area):
    if max_area == 0:
        return 0
    else:
        ca = (max_area - area) / max_area
        return ca
