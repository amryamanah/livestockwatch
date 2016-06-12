import os
import argparse
import re
import asyncio
import shutil
from pprint import pprint
from time import time
import cv2

import numpy as np
import pandas as pd

from IPython import embed

from .pupil_classifier import PupilClassifier
from .descriptors import HogDescriptor
from .pupil_finder import main_detector
from .utils import is_skipped_folder, sort_with_number, write_as_png
from .config import SKIP_FOLDER, DCT_CATTLE_ID
from .manualdata_utils import DataProvider
from .gridregion import GridRegion

def hog_trainer():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dsp", required=True, help="Path to the dataset folder")
    ap.add_argument("-r", "--rp", required=True, help="Path to the dataset folder")
    ap.add_argument("--bk", required=False, type=int, default=15, help="Blur kernel size")
    ap.add_argument("--bt", required=False,
                    choices=['gaussian', 'median', 'bilateral'], default="bilateral", help="Blur type")
    ap.add_argument("--ppc", required=False, type=int, default=4, help="HOG pixels per cell")
    ap.add_argument("--cpb", required=False, type=int, default=1, help="HOG cells per block")
    ap.add_argument("--orientation", required=False, type=int, default=9, help="HOG orientation")
    ap.add_argument("--ct", required=False, default="b_star_hist_equal", help="Color channel type")
    ap.add_argument("--skt", required=False,
                    choices=["poly", "linear", "sigmoid", "rbf"], default="linear",
                    help="SVM kernel type")
    ap.add_argument("--si", required=False,
                    choices=["default", "libsvm", "liblinear"], default="default",
                    help="SVM implementation")

    args = vars(ap.parse_args())
    print(args)
    descriptor = HogDescriptor(
        orientation=args["orientation"], pixels_per_cell=args["ppc"], cells_per_block=args["cpb"]
    )
    svm_classifier = PupilClassifier(
        ds_root_path=args["dsp"], result_folder=args["rp"], channel_type=args["ct"],
        blur_kernel=args["bk"], blur_type=args["bt"],
        descriptor=descriptor, kernel_type=args["skt"], svm_implementation=args["si"]
    )

    svm_classifier.load_dataset()
    svm_classifier.training()


def pupil_extractor():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to the scanned folder")
    ap.add_argument("--svm-path", required=True, help="Path to the scanned folder")
    ap.add_argument("-m", "--main-step", required=False, type=int, default=64, help="Main step")
    ap.add_argument("-s", "--secondary-step", required=False, type=int, default=8, help="Secondary step")
    ap.add_argument("-d", "--duration", required=False, type=float, default=2.0,
                    help="duration of video to be analyzed")
    ap.add_argument("--debug", required=False, type=bool, default=False, help="debug flag")
    ap.add_argument("--bk", required=False, type=int, default=15, help="Blur kernel size")
    ap.add_argument("--bt", required=False,
                    choices=['gaussian', 'median', 'bilateral'], default="bilateral", help="Blur type")
    args = vars(ap.parse_args())
    args["dirpath"] = "something"
    print(args)

    svm_path = os.path.abspath(args["svm_path"])
    svm_path_parts = svm_path.split(os.sep)[-4:]
    svm_kernel_type = svm_path_parts[0]
    hog_info = [int(a) for a in re.findall(r'\d+', svm_path_parts[1])]
    color_channel = svm_path_parts[2]
    scanned_folder = os.path.abspath(args["input"])
    lst_job = []
    lst_cs = [os.path.join(scanned_folder, cs) for cs in os.listdir(scanned_folder)
              if not is_skipped_folder(os.path.join(scanned_folder, cs), SKIP_FOLDER)
              if os.path.isdir(os.path.join(scanned_folder, cs))]

    lst_cs_video = set()
    lst_cs_image = set()

    pprint(lst_cs)
    for cs_dir in lst_cs:
        video_files = list(filename for filename in os.listdir(cs_dir) if filename.endswith(".avi"))
        if len(video_files) > 0:
            lst_cs_video.add(cs_dir)
        else:
            lst_cs_image.add(cs_dir)

    for cs_video in lst_cs_video:
        for filename in os.listdir(cs_video):
            if filename.endswith(".avi"):
                job_kwargs = {
                    "dirpath": cs_video,
                    "filename": filename,
                    "debug": args["debug"],
                    "duration": args["duration"],
                    "main_step": args["main_step"],
                    "secondary_step": args["secondary_step"],
                    "hog_info": hog_info,
                    "svm_kernel_type": svm_kernel_type,
                    "color_channel": color_channel,
                    "svm_path": svm_path,
                    "blur_kernel": args["bk"],
                    "blur_type": args["bt"]
                }
                lst_job.append(job_kwargs)

    for cs_image in lst_cs_image:
        lst_image = []
        for filename in os.listdir(cs_image):
            fn = filename.split(".")[0]
            if filename.endswith(".bmp") and (fn.startswith("nopl") or fn.endswith("no")):
                lst_image.append(filename)
        if len(lst_image[0].split("_")) > 1:
            lst_image = sorted(lst_image)
        else:
            lst_image = sorted(lst_image, key=sort_with_number)

        job_kwargs = {
            "dirpath": cs_image,
            "filename": lst_image,
            "debug": args["debug"],
            "duration": args["duration"],
            "main_step": args["main_step"],
            "secondary_step": args["secondary_step"],
            "hog_info": hog_info,
            "svm_kernel_type": svm_kernel_type,
            "color_channel": color_channel,
            "svm_path": svm_path,
            "blur_kernel": args["bk"],
            "blur_type": args["bt"]
        }
        lst_job.append(job_kwargs)

    start_time = time()
    pprint("[START] Processing {} capture session".format(len(lst_job)))
    for job in lst_job:
        main_detector(**job)

    pprint("[END] Processing {} capture session done in {:.3f} minute".format(
            len(lst_job), (time() - start_time) / 60))
    loop = asyncio.get_event_loop()
    loop.close()


def pupil_manual_extract():
    base_path = "/Volumes/fitramhd/BISE/WA02_month/"
    # lst_excel_path = [os.path.join(base_path, "2015-6-data.xlsx"),]
    lst_excel_path = [os.path.join(base_path, "2015-7-data.xlsx"),
                      os.path.join(base_path, "2015-6-data.xlsx"),
                      os.path.join(base_path, "manual_data.xlsx")]

    lst_plr_df = []
    lst_cs_df = []

    data_provider = DataProvider(excel_result_path="cattle_db.xlsx")

    for excel_path in lst_excel_path:
        plr_df, cs_df = data_provider.process_excel(excel_path)
        if not plr_df.empty:
            lst_plr_df.append(plr_df)
            lst_cs_df.append(cs_df)

    # lst_plr_df.append(data_provider.process_mongo("archive_livestockwatch", "cs1secfirstframe", "old"))
    lst_plr_df.append(data_provider.process_mongo("livestockwatch", "cs1secfirstframe"))
    #
    if lst_plr_df:
        data_provider.save_plr_data(lst_plr_df)
    # #     data_provider.save_cs_data(lst_cs_df)
    #
    data_provider.save_and_process_vita("archive_livestockwatch", "blood_data")


def pupil_region_drawer():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to the scanned folder")
    ap.add_argument("-s", "--subdirname", default="finished", help="subdir name")
    ap.add_argument("-o", "--output", default="nopl", help="name of output folder")
    args = vars(ap.parse_args())

    lst_cat_dir = []
    for root, dirs, files in os.walk(args["input"]):
        for dirname in dirs:
            if dirname == "finished":
                dirpath = os.path.join(root, dirname)
                lst_cat_dir.append(dirpath)

    pprint(lst_cat_dir)

    lst_cs_dir = []
    for cat_dir in lst_cat_dir:
        for cs_dirname in os.listdir(cat_dir):
            cs_dirpath = os.path.join(cat_dir, cs_dirname)
            dirname_parts = cs_dirpath.split(os.sep)
            if not "1111111111" in dirname_parts:
                if os.path.isdir(cs_dirpath):
                    lst_cs_dir.append(cs_dirpath)

    for cs_dir in lst_cs_dir:
        if os.path.isdir(cs_dir):
            grid_region = GridRegion()
            lst_image = []
            for filename in os.listdir(cs_dir):
                fn = filename.split(".")[0]
                if filename.endswith(".bmp") and (fn.startswith("no") or fn.endswith("no")):
                    lst_image.append(filename)
            try:
                if len(lst_image[0].split("_")) > 1:
                    lst_image = sorted(lst_image)
                else:
                    lst_image = sorted(lst_image, key=sort_with_number)
            except Exception:
                embed()

            nopl_result_path = os.path.join(cs_dir, args["output"], "region")

            if os.path.exists(nopl_result_path):
                shutil.rmtree(nopl_result_path)
            os.makedirs(nopl_result_path, exist_ok=True)

            for img_name in lst_image:
                img_path = os.path.join(cs_dir, img_name)
                img = cv2.imread(img_path)

                rows, cols = img.shape[:2]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
                img = cv2.warpAffine(img, M, (cols, rows))
                region_image = grid_region.draw_region(img)
                im_result_name = img_name.split(".")[0] + ".png"
                write_as_png(os.path.join(nopl_result_path, im_result_name), region_image)



