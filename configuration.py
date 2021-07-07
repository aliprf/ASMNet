class DatasetName:
    w300 = '300W'
    wflw = 'wflw'


class ModelArch:
    ASMNet = 'ASMNet'
    MNV2 = 'mobileNetV2'

class DatasetType:
    data_type_train = 0
    data_type_validation = 1
    data_type_test = 2


class LearningConfig:
    batch_size = 3
    epochs = 150
    pose_len = 3


class InputDataSize:
    image_input_size = 224
    pose_len = 3


class W300Conf:
    W300W_prefix_path = '/media/ali/new_data/300W/'  # --> local

    train_pose = W300W_prefix_path + 'train_set/pose/'
    train_annotation = W300W_prefix_path + 'train_set/annotations/'
    train_image = W300W_prefix_path + 'train_set/images/'

    test_annotation_path = W300W_prefix_path + 'test_set/annotations/'
    test_image_path = W300W_prefix_path + 'test_set/images/'
    num_of_landmarks = 68

class WflwConf:
    Wflw_prefix_path = '/media/ali/new_data/wflw/'  # --> local

    train_pose = Wflw_prefix_path + 'train_set/pose/'
    train_annotation = Wflw_prefix_path + 'train_set/annotations/'
    train_image = Wflw_prefix_path + 'train_set/images/'

    test_annotation_path = Wflw_prefix_path + 'test_set/annotations/'
    test_image_path = Wflw_prefix_path + 'test_set/images/'
    num_of_landmarks = 98