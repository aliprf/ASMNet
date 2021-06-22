from configuration import DatasetName, DatasetType, W300Conf, InputDataSize, LearningConfig
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2, l1

from keras.models import Model
from keras.applications import mobilenet_v2
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, GlobalAveragePooling2D, Dropout


class CNNModel:
    def get_model(self, arch, output_len):

        if arch == 'ASMNet':
            model = self.create_ASMNet(inp_shape=[224, 224, 3], output_len=output_len)

        elif arch == 'mobileNetV2':
            model = self.create_mobileNet(inp_shape=[224, 224, 3], output_len=output_len)

        return model

    def create_mobileNet(self, output_len, inp_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   pooling=None)
        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        out_landmarks = Dense(output_len, name='O_L')(x)
        out_poses = Dense(LearningConfig.pose_len, name='O_P')(x)

        inp = mobilenet_model.input
        revised_model = Model(inp, [out_landmarks, out_poses])
        revised_model.summary()
        return revised_model

    def create_ASMNet(self, output_len, inp_tensor=None, inp_shape=None):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=inp_tensor,
                                                   pooling=None)
        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56*56*24
        block_1_project_BN_mpool = GlobalAveragePooling2D()(block_1_project_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28*28*32
        block_3_project_BN_mpool = GlobalAveragePooling2D()(block_3_project_BN)

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14*14*64
        block_6_project_BN_mpool = GlobalAveragePooling2D()(block_6_project_BN)

        block_10_project_BN = mobilenet_model.get_layer('block_10_project_BN').output  # 14*14*96
        block_10_project_BN_mpool = GlobalAveragePooling2D()(block_10_project_BN)

        block_13_project_BN = mobilenet_model.get_layer('block_13_project_BN').output  # 7*7*160
        block_13_project_BN_mpool = GlobalAveragePooling2D()(block_13_project_BN)

        block_15_add = mobilenet_model.get_layer('block_15_add').output  # 7*7*160
        block_15_add_mpool = GlobalAveragePooling2D()(block_15_add)

        x = keras.layers.Concatenate()([block_1_project_BN_mpool, block_3_project_BN_mpool, block_6_project_BN_mpool,
                                        block_10_project_BN_mpool, block_13_project_BN_mpool, block_15_add_mpool])
        x = keras.layers.Dropout(rate=0.3)(x)
        ''''''
        out_landmarks = Dense(output_len,
                              kernel_regularizer=l2(0.01),
                              bias_regularizer=l2(0.01),
                              name='O_L')(x)
        out_poses = Dense(LearningConfig.pose_len,
                          kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01),
                          name='O_P')(x)

        revised_model = Model(inp, [out_landmarks, out_poses])

        revised_model.summary()
        model_json = revised_model.to_json()

        with open("ASMNet.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model