import csv
import datetime
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import savgol_filter
from skimage import color, io
from skimage import restoration, exposure, filters
from IPython import embed


def remove_saturated_value(image):
    image = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (hue, sat, val) = cv2.split(hsv)
    (_, mask) = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
    image = cv2.bitwise_and(image, image, mask=mask)
    return image


def sort_with_number(filename):
    filename = filename.split(".")[0]
    index = int(re.findall(r'\d+', filename)[0])
    return index


def get_datetime_tuple(filename):
    im_path_parts = filename.split("_")
    if len(im_path_parts) > 1:
        datetime_stamps = datetime.datetime(
            int(im_path_parts[1]),
            int(im_path_parts[2]),
            int(im_path_parts[3]),
            int(im_path_parts[4]),
            int(im_path_parts[5]),
            int(im_path_parts[6]),
            int(im_path_parts[7]),
        )
        return datetime_stamps.timestamp()
    else:
        raise ValueError("Wrong format for get_datetime_tuple")


def get_frametimestamp(filename):
    frame_timestamp = None
    filename = filename.split(".")[0]
    frame_number = int(re.findall(r'\d+', filename)[0])
    if frame_number <= 10:
        frame_timestamp = frame_number * 0.1
    elif frame_number >= 11:
        frame_timestamp = 1 + ((frame_number - 10) * 0.2)
    else:
        raise ValueError("Wrong format for get_frametimestamp")
    return frame_timestamp


def svg_smoothing(data_series, window_length=27, polyorder=4, mode="nearest"):
    lst_data_interp = data_series.interpolate(method="linear", limit_direction="both")
    lst_data_savgol = savgol_filter(lst_data_interp, window_length=window_length, polyorder=polyorder, mode=mode)
    return lst_data_savgol


def get_b_star_equalize(image):
    image = filters.gaussian_filter(image, sigma=1, mode='reflect', multichannel=True)
    lab = color.rgb2lab(image)

    return lab[:, :, 2]


def write_csv_result(csv_path, header, data):
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

    with open(csv_path, "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writerow(data)


def write_as_png(newpath, frame):
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    cv2.imwrite(newpath, frame, png_params)


def equ_hist_color_image(image):
    b, g, r = cv2.split(image)
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    return cv2.merge([equ_b, equ_g, equ_r])


def is_skipped_folder(dirpath, skip_folder):
    part_dirpath = dirpath.split(os.sep)
    for x in part_dirpath:
        if x in skip_folder:
            return True
    return False


def plot_confusion_matrix(result_folder, cm, filename, title='Confusion matrix'):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(["negative", "positive"]))
    plt.xticks(tick_marks, ["negative", "positive"], rotation=45)
    plt.yticks(tick_marks, ["negative", "positive"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(result_folder, "{}.png".format(filename)))
    plt.close()


def max_rgb_filter(image):
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    return cv2.merge([B, G, R])


def special_max_r_gb_filter(image):
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    #M = np.maximum(G, B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    green_frame = G.nonzero()
    green_coordinates = [(k, v) for k, v in zip(green_frame[0], green_frame[1])]

    median_blue = np.nanmean(B)
    # median_blue = 255
    for k, v in green_coordinates:
        if B[k,v] == R[k,v] == G[k,v]:
            G[k, v] = 0
            R[k, v] = 0
            B[k, v] = median_blue
        else:
            B[k, v] = G[k, v]
            G[k, v] = 0

    # merge the channels back together and return the image
    return cv2.merge([B, G, R])


def get_specific_channel(image, channel_type, blur_kernel=None, blur_type="gaussian"):

        image = remove_saturated_value(image)
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        image = cv2.medianBlur(image, blur_kernel)
        # if blur_kernel:
        #     if blur_type == "gaussian":
        #         image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        #     elif blur_type == "median":
        #         image = cv2.medianBlur(image, blur_kernel)
        #     elif blur_type == "bilateral":
        #         image = cv2.bilateralFilter(image, blur_kernel, 75, 75)

        if channel_type == "blue":
            blue, green, red = cv2.split(image)
            return blue
        elif channel_type == "green":
            blue, green, red = cv2.split(image)
            return green
        elif channel_type == "red":
            blue, green, red = cv2.split(image)
            return red

        elif channel_type == "hue":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return hue
        elif channel_type == "sat":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return sat
        elif channel_type == "val":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return val

        elif channel_type == "l_star*":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return l_star
        elif channel_type == "a_star*":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return a_star
        elif channel_type == "b_star":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star

        elif channel_type == "b_star_hist_equal":
            b, g, r = cv2.split(image)
            equ_b = cv2.equalizeHist(b)
            equ_g = cv2.equalizeHist(g)
            equ_r = cv2.equalizeHist(r)
            image = cv2.merge([equ_b, equ_g, equ_r])

            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star

        elif channel_type == "gray":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel_type == "gray_hist_equal":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star
        elif channel_type == "chroma":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            chroma = np.sqrt((a*a) + (b*b))
            return chroma
        elif channel_type == "maxrgb_chroma":
            image = max_rgb_filter(image)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            chroma = np.sqrt((a*a) + (b*b))
            return chroma
        elif channel_type == "smrgb_gray":
            image = special_max_r_gb_filter(image)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel_type == "smrgb_hue":
            image = special_max_r_gb_filter(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return hue
        elif channel_type == "maxrgb_gray":
            image = max_rgb_filter(image)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel_type == "maxrgb_hue":
            image = max_rgb_filter(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)
            return hue
        elif channel_type == "maxrgb_b_star":
            image = max_rgb_filter(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_star, a_star, b_star = cv2.split(image)
            return b_star
        else:
            raise "Unsupported channel type {}".format(channel_type)


