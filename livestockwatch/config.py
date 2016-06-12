import os
import pytz


# MONGO_URL = "mongodb://biseadmin:bise2014@162.243.1.142:48203/livestockwatch"

LOCAL_TIMEZONE = pytz.timezone('Asia/Tokyo')
MONGO_URL = "mongodb://localhost:27017"
ORIENTATION = 9
PPC = 4
CPB = 1
KERNEL_TYPE = "linear"
BLUR_KERNEL = 15
BLUR_TYPE = "bilateral"
COLOR_CHANNEL = "b_star_hist_equal"

SVM_RESULT_FOLDER = os.path.join(os.path.abspath(os.getcwd()), "hog_model")

SVM_CLASSIFIER_FOLDER = os.path.join(SVM_RESULT_FOLDER,
                                     "{}".format(KERNEL_TYPE),
                                     "o{}_ppc{}_cpb{}".format(ORIENTATION, PPC, CPB),
                                     "{}".format(COLOR_CHANNEL))

SVM_CLASSIFIER_PATH = os.path.join(SVM_CLASSIFIER_FOLDER, "hog_svm.pkl")

DCT_CATTLE_ID = {
    "WA02": [
        "1445186816", "1445174448", "1445174332",
        "1445184485", "1384106876", "1372243682",
        "1445174745", "860053727"
    ]
}

WIN_SIZE = 256
IMG_WIDTH = 1280
IMG_HEIGHT = 960
IMG_HEIGHT_COS25 = 1059
max_timestamp = 2.0
STEP_SECOND = 1.0

SKIP_FOLDER = ["pending", "finished", "failed", "spare_data",
               "blink_noise", "lightning_noise", "dust_noise", "eyelashes_noise",
               "nopl", "pl",
               "bbox_img", "candidate", "final", "plr_result", "final_extended",
               "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]
