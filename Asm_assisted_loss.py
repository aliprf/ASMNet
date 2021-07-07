import tensorflow as tf
from pca_utility import PCAUtility
import numpy as np


class ASMLoss:
    def __init__(self, dataset_name, accuracy):
        self.dataset_name = dataset_name
        self.accuracy = accuracy

    def calculate_pose_loss(self, x_pr, x_gt):
        return tf.reduce_mean(tf.square(x_gt - x_pr))

    def calculate_landmark_ASM_assisted_loss(self, landmark_pr, landmark_gt, current_epoch, total_steps):
        """
        :param landmark_pr:
        :param landmark_gt:
        :param current_epoch:
        :param total_steps:
        :return:
        """
        # calculating ASMLoss weight:
        asm_weight = 0.5
        if current_epoch < total_steps//3: asm_weight = 2.0
        elif total_steps//3 <= current_epoch < 2*total_steps//3: asm_weight = 1.0

        # creating the ASM-ground truth
        landmark_gt_asm = self._calculate_asm(input_tensor=landmark_gt)

        # calculating ASMLoss
        asm_loss = tf.reduce_mean(tf.square(landmark_gt_asm - landmark_pr))

        # calculating MSELoss
        mse_loss = tf.reduce_mean(tf.square(landmark_gt - landmark_pr))

        # calculating total loss
        return mse_loss + asm_weight * asm_loss

    def _calculate_asm(self, input_tensor):
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(self.dataset_name, pca_percentages=self.accuracy)

        input_vector = np.array(input_tensor)
        out_asm_vector = []
        batch_size = input_vector.shape[0]
        for i in range(batch_size):
            b_vector_p = self._calculate_b_vector(input_vector[i], eigenvalues, eigenvectors, meanvector)
            out_asm_vector.append(meanvector + np.dot(eigenvectors, b_vector_p))

        out_asm_vector = np.array(out_asm_vector)
        return out_asm_vector

    def _calculate_b_vector(self, predicted_vector, eigenvalues, eigenvectors, meanvector):
        b_vector = np.dot(eigenvectors.T, predicted_vector - meanvector)
        # revised b to be in -3lambda =>
        i = 0
        for b_item in b_vector:
            lambda_i_sqr = 3 * np.sqrt(eigenvalues[i])
            if b_item > 0:
                b_item = min(b_item, lambda_i_sqr)
            else:
                b_item = max(b_item, -1 * lambda_i_sqr)
            b_vector[i] = b_item
            i += 1

        return b_vector



