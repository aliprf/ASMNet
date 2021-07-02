from configuration import DatasetName, WflwConf, W300Conf, DatasetType, LearningConfig, InputDataSize
import tensorflow as tf

import cv2
import os.path
import scipy.io as sio
from cnn_model import CNNModel
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.integrate import simps
from scipy.integrate import trapz
import matplotlib.pyplot as plt

class Test:
    def test_model(self, pretrained_model_path, ds_name):
        if ds_name == DatasetName.w300:
            test_annotation_path = W300Conf.test_annotation_path
            test_image_path = W300Conf.test_image_path
        elif ds_name == DatasetName.wflw:
            test_annotation_path = WflwConf.test_annotation_path
            test_image_path = WflwConf.test_image_path

        model = tf.keras.models.load_model(pretrained_model_path)

        for i, file in tqdm(enumerate(os.listdir(test_image_path))):
            img = test_image_path + file
            model.predict(np.expand_dims(img, axis=0))

            landmark_predicted = model.predict[0][0]
            pose_predicted = model.predict[1][0]
