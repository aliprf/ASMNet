from configuration import DatasetName, WflwConf, W300Conf, DatasetType, LearningConfig, InputDataSize
from cnn_model import CNNModel
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import csv
from skimage.io import imread
import pickle
from image_utility import ImageUtility
from tqdm import tqdm
import os
from Asm_assisted_loss import ASMLoss
from cnn_model import CNNModel


class Train:
    def __init__(self, arch, dataset_name, save_path, asm_accuracy=90):
        """
        :param arch:
        :param dataset_name:
        :param save_path:
        :param asm_accuracy:
        """

        self.dataset_name = dataset_name
        self.save_path = save_path
        self.arch = arch
        self.asm_accuracy = asm_accuracy

        if dataset_name == DatasetName.w300:
            self.num_landmark = W300Conf.num_of_landmarks * 2
            self.img_path = W300Conf.train_image
            self.annotation_path = W300Conf.train_annotation
            self.pose_path = W300Conf.train_pose

        if dataset_name == DatasetName.wflw:
            self.num_landmark = WflwConf.num_of_landmarks * 2
            self.img_path = WflwConf.train_image
            self.annotation_path = WflwConf.train_annotation
            self.pose_path = WflwConf.train_pose

    def train(self, weight_path):
        """

        :param weight_path:
        :return:
        """

        '''create loss'''
        c_loss = ASMLoss(dataset_name=self.dataset_name, accuracy=90)
        cnn = CNNModel()
        '''making models'''
        model = cnn.get_model(arch=self.arch, output_len=self.num_landmark)
        if weight_path is not None:
            model.load_weights(weight_path)

        '''create sample generator'''
        image_names, landmark_names, pose_names = self._create_generators()

        '''create train configuration'''
        step_per_epoch = len(image_names) // LearningConfig.batch_size

        '''start train:'''
        optimizer = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-5)
        for epoch in range(LearningConfig.epochs):
            image_names, landmark_names, pose_names = shuffle(image_names, landmark_names, pose_names)
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, annotation_gr, poses_gr = self._get_batch_sample(
                    batch_index=batch_index,
                    img_filenames=image_names,
                    landmark_filenames=landmark_names,
                    pose_filenames=pose_names)

                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                annotation_gr = tf.cast(annotation_gr, tf.float32)
                poses_gr = tf.cast(poses_gr, tf.float32)

                '''train step'''
                self.train_step(epoch=epoch,
                                step=batch_index,
                                total_steps=step_per_epoch,
                                model=model,
                                images=images,
                                annotation_gt=annotation_gr,
                                poses_gt=poses_gr,
                                optimizer=optimizer,
                                c_loss=c_loss)
            '''save weights'''
            model.save(self.save_path + self.arch + str(epoch) + '_' + self.dataset_name)

    def train_step(self, epoch, step, total_steps, model, images, annotation_gt, poses_gt, optimizer, c_loss):
        """

        :param epoch:
        :param step:
        :param total_steps:
        :param model:
        :param images:
        :param annotation_gt:
        :param poses_gt:
        :param optimizer:
        :param c_loss:
        :return:
        """

        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            annotation_predicted, pose_predicted = model(images, training=True)
            '''calculate loss'''
            mse_loss, asm_loss = c_loss.calculate_landmark_ASM_assisted_loss(landmark_pr=annotation_predicted,
                                                                             landmark_gt=annotation_gt,
                                                                             current_epoch=epoch,
                                                                             total_steps=total_steps)
            pose_loss = c_loss.calculate_pose_loss(x_pr=pose_predicted, x_gt=poses_gt)

            '''calculate loss'''
            total_loss = mse_loss + asm_loss + pose_loss

        '''calculate gradient'''
        gradients_of_model = tape.gradient(total_loss, model.trainable_variables)
        '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps), ' -> : total_loss: ',
                 total_loss)

    def _create_generators(self):
        """
        :return:
        """
        image_names, landmark_filenames, pose_names = \
            self._create_image_and_labels_name(img_path=self.img_path,
                                               annotation_path=self.annotation_path,
                                               pose_path=self.pose_path)
        return image_names, landmark_filenames, pose_names

    def _create_image_and_labels_name(self, img_path, annotation_path, pose_path):
        """

        :param img_path:
        :param annotation_path:
        :param pose_path:
        :return:
        """
        img_filenames = []
        landmark_filenames = []
        poses_filenames = []

        for file in os.listdir(img_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                lbl_file = str(file)[:-3] + "npy"  # just name
                pose_file = str(file)[:-3] + "npy"  # just name
                if os.path.exists(annotation_path + lbl_file) and os.path.exists(pose_path + lbl_file):
                    img_filenames.append(str(file))
                    landmark_filenames.append(lbl_file)
                    poses_filenames.append(pose_file)

        return np.array(img_filenames), np.array(landmark_filenames), np.array(poses_filenames)

    def _get_batch_sample(self, batch_index, img_filenames, landmark_filenames, pose_filenames):
        """
        :param batch_index:
        :param img_filenames:
        :param landmark_filenames:
        :param pose_filenames:
        :return:
        """

        '''create batch data and normalize images'''
        batch_img = img_filenames[
                    batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_lnd = landmark_filenames[
                    batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_pose = pose_filenames[
                     batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        '''create img and annotations'''
        img_batch = np.array([imread(self.img_path + file_name) for file_name in batch_img]) / 255.0
        lnd_batch = np.array([self._load_and_normalize(self.annotation_path + file_name) for file_name in batch_lnd])
        pose_batch = np.array([load(self.pose_path + file_name) for file_name in batch_pose])

        return img_batch, lnd_batch, pose_batch

    def _load_and_normalize(self, point_path):
        """
        :param point_path:
        :return:
        """

        annotation = load(point_path)
        '''normalize landmarks'''
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm
