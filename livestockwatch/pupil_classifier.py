import os
import logging
from time import time
import csv
import pprint

import cv2
import numpy as np


from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.externals import joblib

from .utils import write_csv_result, get_specific_channel

from IPython import embed

logger = logging.getLogger(__name__)


class PupilClassifier:
    def __init__(self, ds_root_path, result_folder,
                 channel_type, blur_kernel, blur_type, descriptor,
                 kernel_type, svm_implementation="default"):

        self.dataset = None
        self.dataset_label = None
        self.train_dataset = None
        self.train_dataset_label = None
        self.test_dataset = None
        self.test_dataset_label = None

        self.ds_root_path = ds_root_path
        self.descriptor = descriptor
        self.channel_type = channel_type
        self.win_size = 256
        self.kernel_type = kernel_type
        assert self.kernel_type in ["poly", "linear", "sigmoid", "rbf"], \
            "Unsupported kernel type = {}".format(kernel_type)

        self.svm_implementation = svm_implementation
        assert self.svm_implementation in ["default", "libsvm", "liblinear"], \
            "Unsupported svm implementation = {}".format(svm_implementation)

        if self.svm_implementation == "liblinear":
            assert self.kernel_type == "linear", "Lib linear can only support Linear kernel type"

        self.blur_kernel = blur_kernel
        self.blur_type = blur_type
        self.result_folder = result_folder
        if self.descriptor.__str__() == "HOG":
            self.svm_classifier_output_folder = os.path.join(self.result_folder,
                                                             self.kernel_type,
                                                             "o{}_ppc{}_cpb{}".format(
                                                                     self.descriptor.orientation,
                                                                     self.descriptor.pixels_per_cell,
                                                                     self.descriptor.cells_per_block
                                                             ),
                                                             "{}".format(self.channel_type))

            os.makedirs(self.svm_classifier_output_folder, exist_ok=True)
            self.svm_classifier_path = os.path.join(self.svm_classifier_output_folder, "hog_svm.pkl")

    def extract_patch(self, img, point):
        x = int(point[0])
        y = int(point[1])
        patch = img[y:y+self.win_size, x:x+self.win_size]
        assert patch.shape[:2] == (self.win_size, self.win_size), "shape is not correct"

        return patch

    def load_dataset(self):
        t0 = time()
        logger.info("[START] Load data set ")

        nopl_count = 0
        nopl_data = []
        nopl_label = []

        # pl_count = 0
        # pl_data = []
        # pl_label = []

        neg_count = 0
        neg_data = []
        neg_label = []

        pl_annotation_dict = {}
        nopl_annotation_dict = {}

        for label_name in ["nopl_{}".format(self.win_size),
                           "pl_{}".format(self.win_size)]:
            annotation_path = os.path.join(self.ds_root_path, label_name, "annotations.csv")
            if os.path.exists(annotation_path):
                with open(annotation_path, newline='\n') as csvfile:
                    annotation_reader = csv.reader(csvfile)
                    for row in annotation_reader:
                        if row[0] == "filename":
                            continue
                        else:
                            if label_name == "pl_{}".format(self.win_size):
                                pl_annotation_dict[row[0]] = row[1], row[2]
                            elif label_name == "nopl_{}".format(self.win_size):
                                nopl_annotation_dict[row[0]] = row[1], row[2]

        for dirpath, dirnames, files in os.walk(self.ds_root_path):
            for filename in files:
                if filename.endswith(".bmp"):
                    label_name = os.path.basename(dirpath)
                    image = cv2.imread(os.path.join(dirpath, filename))
                    gray = get_specific_channel(image, self.channel_type,
                                                blur_kernel=self.blur_kernel, blur_type=self.blur_type)

                    # preview = image.copy()
                    #
                    # if not preview.shape == (self.win_size, self.win_size, 3):
                    #     preview = self.extract_patch(preview, nopl_annotation_dict[filename])
                    #
                    #     b, g, r = cv2.split(preview)
                    #     equ_b = cv2.equalizeHist(b)
                    #     equ_g = cv2.equalizeHist(g)
                    #     equ_r = cv2.equalizeHist(r)
                    #     final_patch_equ = cv2.merge([equ_b, equ_g, equ_r])
                    #
                    #     cv2.imshow("nopl", final_patch_equ)
                    #     cv2.waitKey(0)
                    #
                    # # delete_flag = int(input("Delete ?(0,1) "))
                    # # if delete_flag:
                    # #     print("deleting {}".format(os.path.join(dirpath, filename)))
                    # #     os.remove(os.path.join(dirpath, filename))

                    if label_name == "neg_{}".format(self.win_size):
                        assert gray.shape == (256, 256)
                        img_feature = self.descriptor.describe(gray)
                        neg_data.append(img_feature)
                        neg_label.append(0)
                        neg_data[neg_count] = img_feature
                        neg_label[neg_count] = 0
                        neg_count += 1

                    elif label_name == "nopl_{}".format(self.win_size):
                        if not gray.shape == (self.win_size, self.win_size):
                            gray = self.extract_patch(gray, nopl_annotation_dict[filename])
                        assert gray.shape == (256, 256)

                        img_feature = self.descriptor.describe(gray)
                        nopl_data.append(img_feature)
                        nopl_label.append(1)
                        nopl_count += 1
                    # elif label_name == "pl_{}".format(self.win_size):
                    #     if not gray.shape == (self.win_size, self.win_size):
                    #         gray = self.extract_patch(gray, pl_annotation_dict[filename])
                    #     assert gray.shape == (256, 256)
                    #     img_feature = self.descriptor.describe(gray)
                    #     pl_data.append(img_feature)
                    #     pl_label.append(1)
                    #     pl_count += 1

        neg_data = np.array(neg_data)
        neg_label = np.array(neg_label)
        nopl_data = np.array(nopl_data)
        nopl_label = np.array(nopl_label)
        # pl_data = np.array(pl_data)
        # pl_label = np.array(pl_label)
        # self.dataset = np.concatenate((pl_data, nopl_data, neg_data))
        # self.dataset_label = np.concatenate((pl_label, nopl_label, neg_label))

        self.dataset = np.concatenate((nopl_data, neg_data))
        self.dataset_label = np.concatenate((nopl_label, neg_label))

        (self.train_dataset, self.test_dataset, self.train_dataset_label, self.test_dataset_label) = train_test_split(
            np.array(self.dataset), self.dataset_label, test_size=0.2, random_state=42
        )

        logger.info("done in %0.3fs" % (time() - t0))
        logger.info("[START] Finish loading data set (train = {}, testing = {})".format(
            self.train_dataset.shape, self.test_dataset.shape
        ))

    def training(self):
        kfold_num = 5
        param_grid = {}
        perf_data = {
            "svm_implementation": self.svm_implementation,
            "svm_kernel_type": self.kernel_type,
            "orientation": self.descriptor.orientation,
            'ppc': self.descriptor.pixels_per_cell,
            "cpb": self.descriptor.cells_per_block,
            "color_channel": self.channel_type,
            "blur_type": self.blur_type,
            "blur_kernel": self.blur_kernel,
            'cv_kfold_num': kfold_num,
            'prec_neg': None, 'prec_pos': None, 'prec_weight_avg': None,
            'recall_neg': None, 'recall_pos': None, 'recall_weight_avg': None,
            'f1_neg': None, 'f1_pos': None, 'f1_weight_avg': None,
            'test_data_pos': None, 'test_data_neg': None, 'test_data_tot': None
        }

        perf_header = [
            "svm_implementation", "svm_kernel_type", "orientation", "ppc", "cpb",
            "color_channel","blur_type", "blur_kernel",
            "cv_kfold_num",
            "prec_neg", "prec_pos", "prec_weight_avg",
            "recall_neg", "recall_pos", "recall_weight_avg",
            "f1_neg", "f1_pos", "f1_weight_avg",
            "test_data_pos", "test_data_neg", "test_data_tot"
        ]

        logger.info('[Start] Train SVM with kernel_type = {}'.format(self.kernel_type))
        start_time = time()
        # Train a SVM classification model
        logger.info("Fitting the classifier to the training set")
        t0 = time()

        if self.svm_implementation in ["default", "liblinear"]:
            param_grid["C"] = np.logspace(-3, 2, 6)
        elif self.svm_implementation == "libsvm":
            param_grid["nu"] = np.arange(0.1, 0.9, 0.1)

        if self.kernel_type in ["poly", "rbf", "sigmoid"]:
            param_grid["gamma"] = np.logspace(-3, 2, 6)

        if self.kernel_type == "poly":
            param_grid["degree"] = np.arange(3, 6)

        cv = cross_validation.StratifiedKFold(self.train_dataset_label, n_folds=kfold_num,
                                              shuffle=False, random_state=None)

        if self.svm_implementation == "default":
            svm_classification = GridSearchCV(SVC(kernel=self.kernel_type,
                                                  class_weight='balanced', cache_size=1000,
                                                  verbose=False, decision_function_shape="ovr"),
                                              param_grid, cv=cv)

        elif self.svm_implementation == "libsvm":
            svm_classification = GridSearchCV(NuSVC(kernel=self.kernel_type,
                                                    class_weight='balanced', cache_size=1000,
                                                    verbose=False, decision_function_shape="ovr"),
                                              param_grid, cv=cv)

        elif self.svm_implementation == "liblinear":
            svm_classification = GridSearchCV(LinearSVC(class_weight='balanced', verbose=False),
                                              param_grid, cv=cv)

        else:
            raise "Unsupported svm_implementation = {}".format(self.svm_implementation)

        svm_classification = svm_classification.fit(self.train_dataset, self.train_dataset_label)

        logger.info("done in %0.3fs" % (time() - t0))
        logger.info("Best estimator found by grid search:")
        logger.info(svm_classification.best_estimator_)

        ###############################################################################
        # Quantitative evaluation of the model quality on the test set

        logger.info("Predicting Cattle Eye on the test set")
        t0 = time()
        y_pred = svm_classification.predict(self.test_dataset)
        logger.info("done in %0.3fs" % (time() - t0))

        labels = unique_labels(self.test_dataset_label, y_pred)
        p, r, f1, s = metrics.precision_recall_fscore_support(self.test_dataset_label, y_pred, labels=labels,
                                                              average=None, sample_weight=None)

        perf_data["prec_neg"] = p[0]
        perf_data["prec_pos"] = p[1]
        perf_data["prec_weight_avg"] = ((p[0] * s[0]) + (p[1] * s[1])) / np.sum(s)
        perf_data["recall_neg"] = r[0]
        perf_data["recall_pos"] = r[1]
        perf_data["recall_weight_avg"] = ((r[0] * s[0]) + (r[1] * s[1])) / np.sum(s)
        perf_data["f1_neg"] = f1[0]
        perf_data["f1_pos"] = f1[1]
        perf_data["f1_weight_avg"] = ((f1[0] * s[0]) + (f1[1] * s[1])) / np.sum(s)
        perf_data["test_data_neg"] = s[0]
        perf_data["test_data_pos"] = s[1]
        perf_data["test_data_tot"] = np.sum(s)

        logger.info("Classification result")
        logger.info(pprint.pformat(perf_data))
        logger.info("[END] Train SVM with done in %0.3fs" % (time() - start_time))
        write_csv_result(os.path.join(self.result_folder, "report.csv"), perf_header, perf_data)
        joblib.dump(svm_classification, self.svm_classifier_path)
