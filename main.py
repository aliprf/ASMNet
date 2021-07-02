from train import Train
from test import Test
from configuration import DatasetName
if __name__ == '__main__':

    '''use the pretrained model'''
    tester = Test()
    tester.test_model(ds_name=DatasetName.w300,
                      pretrained_model_path='./pre_trained_models/ASMNet/ASM_loss/ASMNet_300W_ASMLoss.h5')
