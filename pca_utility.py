from configuration import DatasetName, DatasetType, W300Conf,\
    InputDataSize, LearningConfig, WflwConf, CofwConf
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

    def create_pca_from_npy(self, dataset_name, pca_postfix):
        lbl_arr = []
        path = None
        if dataset_name == DatasetName.ibug:
            path = W300Conf.normalized_points_npy_dir  # normalized
        elif dataset_name == DatasetName.cofw:
            path = CofwConf.normalized_points_npy_dir  # normalized
        elif dataset_name == DatasetName.wflw:
            path = WflwConf.normalized_points_npy_dir  # normalized

        print('PCA calculation started: loading labels')

        lbl_arr = []
        for file in tqdm(os.listdir(path)):
            if file.endswith(".npy"):
                npy_file = os.path.join(path, file)
                lbl_arr.append(load(npy_file))

        lbl_arr = np.array(lbl_arr)

        ''' no normalization is needed, since we want to generate hm'''
        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA(lbl_arr, pca_postfix)
        mean_lbl_arr = np.mean(lbl_arr, axis=0)
        eigenvectors = eigenvectors.T
        #
        # self.__save_obj(eigenvalues, dataset_name + self.__eigenvalues_prefix + str(pca_postfix))
        # self.__save_obj(eigenvectors, dataset_name + self.__eigenvectors_prefix + str(pca_postfix))
        # self.__save_obj(mean_lbl_arr, dataset_name + self.__meanvector_prefix + str(pca_postfix))
        #
        save('pca_obj/' + dataset_name + self.eigenvalues_prefix + str(pca_postfix), eigenvalues)
        save('pca_obj/' + dataset_name + self.eigenvectors_prefix + str(pca_postfix), eigenvectors)
        save('pca_obj/' + dataset_name + self.meanvector_prefix + str(pca_postfix), mean_lbl_arr)

    def calculate_b_vector(self, predicted_vector, correction, eigenvalues, eigenvectors, meanvector):
        tmp1 = predicted_vector - meanvector
        b_vector = np.dot(eigenvectors.T, tmp1)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    b_item = max(b_item, -1 * lambda_i_sqr)
                b_vector[i] = b_item
                i += 1

        return b_vector


    def _func_PCA(self, input_data, pca_postfix):
        input_data = np.array(input_data)
        pca = PCA(n_components=pca_postfix/100)
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


