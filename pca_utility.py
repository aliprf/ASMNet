from configuration import DatasetName, DatasetType, W300Conf, InputDataSize, LearningConfig, WflwConf
from image_utility import ImageUtility
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle
import os
from tqdm import tqdm
from numpy import save, load
import math
from PIL import Image
from numpy import save, load


class PCAUtility:
    eigenvalues_prefix = "_eigenvalues_"
    eigenvectors_prefix = "_eigenvectors_"
    meanvector_prefix = "_meanvector_"



    def create_pca_from_npy(self, dataset_name, labels_npy_path, pca_percentages):
        """
        generate and save eigenvalues, eigenvectors, meanvector
        :param labels_npy_path: the path to the normalized labels that are save in npy format.
        :param pca_percentages: % of eigenvalues that will be used
        :return: generate
        """
        path = labels_npy_path
        print('PCA calculation started: loading labels')

        lbl_arr = []
        for file in tqdm(os.listdir(path)):
            if file.endswith(".npy"):
                npy_file = os.path.join(path, file)
                lbl_arr.append(load(npy_file))

        lbl_arr = np.array(lbl_arr)

        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA(lbl_arr, pca_percentages)
        mean_lbl_arr = np.mean(lbl_arr, axis=0)
        eigenvectors = eigenvectors.T

        save('./pca_obj/' + dataset_name + self.eigenvalues_prefix + str(pca_percentages), eigenvalues)
        save('./pca_obj/' + dataset_name + self.eigenvectors_prefix + str(pca_percentages), eigenvectors)
        save('./pca_obj/' + dataset_name + self.meanvector_prefix + str(pca_percentages), mean_lbl_arr)

    def load_pca_obj(self, dataset_name, pca_percentages):
        eigenvalues = np.load('./pca_obj/' + dataset_name + self.eigenvalues_prefix + str(pca_percentages))
        eigenvectors = np.load('./pca_obj/' + dataset_name + self.eigenvectors_prefix + str(pca_percentages))
        meanvector = np.load('./pca_obj/' + dataset_name + self.meanvector_prefix + str(pca_percentages))
        return eigenvalues, eigenvectors, meanvector

    def _func_PCA(self, input_data, pca_postfix):
        input_data = np.array(input_data)
        pca = PCA(n_components=pca_postfix / 100)
        # pca = PCA(n_components=0.98)
        # pca = IncrementalPCA(n_components=50, batch_size=50)
        pca.fit(input_data)
        pca_input_data = pca.transform(input_data)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        return pca_input_data, eigenvalues, eigenvectors

    def __svd_func(self, input_data, pca_postfix):
        svd = TruncatedSVD(n_components=50)
        svd.fit(input_data)
        pca_input_data = svd.transform(input_data)
        eigenvalues = svd.explained_variance_
        eigenvectors = svd.components_
        return pca_input_data, eigenvalues, eigenvectors
        # U, S, VT = svd(input_data)
