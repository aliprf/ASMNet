import random
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import math
from skimage.transform import warp, AffineTransform
import cv2
from scipy import misc
from skimage.transform import rotate
from PIL import Image
from PIL import ImageOps
from skimage.transform import resize
from skimage import transform
from skimage.transform import SimilarityTransform, AffineTransform
import random
from configuration import  DatasetName

class ImageUtility:

    def crop_and_save(self, _image, _label, file_name, num_of_landmarks, dataset_name):
        try:
            '''crop data: we add a small margin to the images'''

            xy_points, x_points, y_points = self.create_landmarks(landmarks=_label,
                                                                      scale_factor_x=1, scale_factor_y=1)

            # self.print_image_arr(str(x_points[0]), _image, x_points, y_points)

            img_arr, points_arr = self.cropImg(_image, x_points, y_points, no_padding=False)
            # img_arr = output_img
            # points_arr = t_label
            '''resize image to 224*224'''
            resized_img = resize(img_arr,
                                 (224, 224, 3),
                                 anti_aliasing=True)
            dims = img_arr.shape
            height = dims[0]
            width = dims[1]
            scale_factor_y = 224 / height
            scale_factor_x = 224 / width

            '''rescale and retrieve landmarks'''
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = \
                self.create_landmarks(landmarks=points_arr,
                                      scale_factor_x=scale_factor_x,
                                      scale_factor_y=scale_factor_y)

            min_b = 0.0
            max_b = 224
            if not(min(landmark_arr_x) < min_b or min(landmark_arr_y) < min_b or
                   max(landmark_arr_x) > max_b or max(landmark_arr_y) > max_b):

                # self.print_image_arr(str(landmark_arr_x[0]), resized_img, landmark_arr_x, landmark_arr_y)

                im = Image.fromarray((resized_img * 255).astype(np.uint8))
                im.save(str(file_name) + '.jpg')

                pnt_file = open(str(file_name) + ".pts", "w")
                pre_txt = ["version: 1 \n", "n_points: 68 \n", "{ \n"]
                pnt_file.writelines(pre_txt)
                points_txt = ""
                for i in range(0, len(landmark_arr_xy), 2):
                    points_txt += str(landmark_arr_xy[i]) + " " + str(landmark_arr_xy[i + 1]) + "\n"

                pnt_file.writelines(points_txt)
                pnt_file.write("} \n")
                pnt_file.close()

        except Exception as e:
            print(e)

    def random_rotate(self, _image, _label, file_name, num_of_landmarks, dataset_name):
        try:

            xy_points, x_points, y_points = self.create_landmarks(landmarks=_label,
                                                                  scale_factor_x=1, scale_factor_y=1)
            # self.print_image_arr(str(xy_points[8]), _image, x_points, y_points)

            _image, _label = self.cropImg_2time(_image, x_points, y_points)

            _image = self.__noisy(_image)

            scale = (np.random.uniform(0.8, 1.0), np.random.uniform(0.8, 1.0))
            # scale = (1, 1)

            rot = np.random.uniform(-1 * 0.55, 0.55)
            translation = (0, 0)
            shear = 0

            tform = AffineTransform(
                scale=scale,  # ,
                rotation=rot,
                translation=translation,
                shear=np.deg2rad(shear)
            )

            output_img = transform.warp(_image, tform.inverse, mode='symmetric')

            sx, sy = scale
            t_matrix = np.array([
                [sx * math.cos(rot), -sy * math.sin(rot + shear), 0],
                [sx * math.sin(rot), sy * math.cos(rot + shear), 0],
                [0, 0, 1]
            ])
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(_label, 1, 1)
            label = np.array(landmark_arr_x + landmark_arr_y).reshape([2, num_of_landmarks])
            marging = np.ones([1, num_of_landmarks])
            label = np.concatenate((label, marging), axis=0)

            label_t = np.dot(t_matrix, label)
            lbl_flat = np.delete(label_t, 2, axis=0).reshape([2*num_of_landmarks])

            t_label = self.__reorder(lbl_flat, num_of_landmarks)

            '''crop data: we add a small margin to the images'''
            xy_points, x_points, y_points = self.create_landmarks(landmarks=t_label,
                                                                  scale_factor_x=1, scale_factor_y=1)
            img_arr, points_arr = self.cropImg(output_img, x_points, y_points, no_padding=False)
            # img_arr = output_img
            # points_arr = t_label
            '''resize image to 224*224'''
            resized_img = resize(img_arr,
                                 (224, 224, 3),
                                 anti_aliasing=True)
            dims = img_arr.shape
            height = dims[0]
            width = dims[1]
            scale_factor_y = 224 / height
            scale_factor_x = 224 / width

            '''rescale and retrieve landmarks'''
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = \
                self.create_landmarks(landmarks=points_arr,
                                      scale_factor_x=scale_factor_x,
                                      scale_factor_y=scale_factor_y)

            min_b = 0.0
            max_b = 224
            if dataset_name == DatasetName.cofw:
                min_b = 5.0
                max_b = 214

            if not(min(landmark_arr_x) < 0 or min(landmark_arr_y) < min_b or
                   max(landmark_arr_x) > 224 or max(landmark_arr_y) > max_b):

                # self.print_image_arr(str(landmark_arr_x[0]), resized_img, landmark_arr_x, landmark_arr_y)

                im = Image.fromarray((resized_img * 255).astype(np.uint8))
                im.save(str(file_name) + '.jpg')

                pnt_file = open(str(file_name) + ".pts", "w")
                pre_txt = ["version: 1 \n", "n_points: 68 \n", "{ \n"]
                pnt_file.writelines(pre_txt)
                points_txt = ""
                for i in range(0, len(landmark_arr_xy), 2):
                    points_txt += str(landmark_arr_xy[i]) + " " + str(landmark_arr_xy[i + 1]) + "\n"

                pnt_file.writelines(points_txt)
                pnt_file.write("} \n")
                pnt_file.close()

            return t_label, output_img
        except Exception as e:
            print(e)
            return None, None


    def random_rotate_m(self, _image, _label_img, file_name):

        rot = random.uniform(-80.9, 80.9)

        output_img = rotate(_image, rot, resize=True)
        output_img_lbl = rotate(_label_img, rot, resize=True)

        im = Image.fromarray((output_img * 255).astype(np.uint8))
        im_lbl = Image.fromarray((output_img_lbl * 255).astype(np.uint8))

        im_m = ImageOps.mirror(im)
        im_lbl_m = ImageOps.mirror(im_lbl)

        im.save(str(file_name)+'.jpg')
        # im_lbl.save(str(file_name)+'_lbl.jpg')

        im_m.save(str(file_name) + '_m.jpg')
        # im_lbl_m.save(str(file_name) + '_m_lbl.jpg')

        im_lbl_ar = np.array(im_lbl)
        im_lbl_m_ar = np.array(im_lbl_m)

        self.__save_label(im_lbl_ar, file_name, np.array(im))
        self.__save_label(im_lbl_m_ar, file_name+"_m", np.array(im_m))


    def __save_label(self, im_lbl_ar, file_name, img_arr):

        im_lbl_point = []
        for i in range(im_lbl_ar.shape[0]):
            for j in range(im_lbl_ar.shape[1]):
                if im_lbl_ar[i, j] != 0:
                    im_lbl_point.append(j)
                    im_lbl_point.append(i)

        pnt_file = open(str(file_name)+".pts", "w")

        pre_txt = ["version: 1 \n", "n_points: 68 \n", "{ \n"]
        pnt_file.writelines(pre_txt)
        points_txt = ""
        for i in range(0, len(im_lbl_point), 2):
            points_txt += str(im_lbl_point[i]) + " " + str(im_lbl_point[i+1]) + "\n"

        pnt_file.writelines(points_txt)
        pnt_file.write("} \n")
        pnt_file.close()

        '''crop data: we add a small margin to the images'''
        xy_points, x_points, y_points = self.create_landmarks(landmarks=im_lbl_point,
                                                                       scale_factor_x=1, scale_factor_y=1)
        img_arr, points_arr = self.cropImg(img_arr, x_points, y_points)

        '''resize image to 224*224'''
        resized_img = resize(img_arr,
                             (224, 224, 3),
                             anti_aliasing=True)
        dims = img_arr.shape
        height = dims[0]
        width = dims[1]
        scale_factor_y = 224 / height
        scale_factor_x = 224 / width

        '''rescale and retrieve landmarks'''
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = \
            self.create_landmarks(landmarks=points_arr,
                                           scale_factor_x=scale_factor_x,
                                           scale_factor_y=scale_factor_y)

        im = Image.fromarray((resized_img * 255).astype(np.uint8))
        im.save(str(im_lbl_point[0])+'.jpg')
        # self.print_image_arr(im_lbl_point[0], resized_img, landmark_arr_x, landmark_arr_y)


    def augment(self, _image, _label, num_of_landmarks):

        # face = misc.face(gray=True)
        #
        # rotate_face = ndimage.rotate(_image, 45)
        # self.print_image_arr(_label[0], rotate_face, [],[])

        # hue_img = tf.image.random_hue(_image, max_delta=0.1)  # max_delta must be in the interval [0, 0.5].
        # sat_img = tf.image.random_saturation(hue_img, lower=0.0, upper=3.0)
        #
        # sat_img = K.eval(sat_img)
        #
        _image = self.__noisy(_image)

        shear = 0

        # rot = 0.0
        '''this scale has problem'''
        # scale = (random.uniform(0.8, 1.00), random.uniform(0.8, 1.00))

        scale = (1, 1)

        rot = np.random.uniform(-1 * 0.008, 0.008)

        tform = AffineTransform(scale=scale, rotation=rot, shear=shear,
                                translation=(0, 0))

        output_img = warp(_image, tform.inverse, output_shape=(_image.shape[0], _image.shape[1]))

        sx, sy = scale
        t_matrix = np.array([
            [sx * math.cos(rot), -sy * math.sin(rot + shear), 0],
            [sx * math.sin(rot), sy * math.cos(rot + shear), 0],
            [0, 0, 1]
        ])
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = self.create_landmarks(_label, 1, 1)
        label = np.array(landmark_arr_x + landmark_arr_y).reshape([2, num_of_landmarks])
        marging = np.ones([1, num_of_landmarks])
        label = np.concatenate((label, marging), axis=0)

        label_t = np.dot(t_matrix, label)
        lbl_flat = np.delete(label_t, 2, axis=0).reshape([num_of_landmarks*2])

        t_label = self.__reorder(lbl_flat, num_of_landmarks)
        return t_label, output_img

    def __noisy(self, image):
        noise_typ = random.randint(0, 5)
        # if True or noise_typ == 0 :#"gauss":
        #     row, col, ch = image.shape
        #     mean = 0
        #     var = 0.001
        #     sigma = var ** 0.1
        #     gauss = np.random.normal(mean, sigma, (row, col, ch))
        #     gauss = gauss.reshape(row, col, ch)
        #     noisy = image + gauss
        #     return noisy
        if 1 <= noise_typ <= 2:# "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out

        # elif 5 <=noise_typ <= 7: #"speckle":
        #     row, col, ch = image.shape
        #     gauss = np.random.randn(row, col, ch)
        #     gauss = gauss.reshape(row, col, ch)
        #     noisy = image + image * gauss
        #     return noisy
        else:
            return image

    def __reorder(self, input_arr, num_of_landmarks):
        out_arr = []
        for i in range(num_of_landmarks):
            out_arr.append(input_arr[i])
            k = num_of_landmarks + i
            out_arr.append(input_arr[k])
        return np.array(out_arr)

    def print_image_arr_heat(self, k, image):
        plt.figure()
        plt.imshow(image)
        implot = plt.imshow(image)
        plt.axis('off')
        plt.savefig('heat' + str(k) + '.png', bbox_inches='tight')
        plt.clf()

    def print_image_arr(self, k, image, landmarks_x, landmarks_y):
        plt.figure()
        plt.imshow(image)
        implot = plt.imshow(image)

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='black', s=20)
        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='white', s=15)
        plt.axis('off')
        plt.savefig('sss' + str(k) + '.png',  bbox_inches='tight')
        # plt.show()
        plt.clf()

    def create_landmarks_from_normalized_original_img(self, img, landmarks, width, height, x_center, y_center, x1, y1, scale_x, scale_y):
        # landmarks_splited = _landmarks.split(';')
        landmark_arr_xy = []
        landmark_arr_x = []
        landmark_arr_y = []

        for j in range(0, len(landmarks), 2):
            x = ((x_center - float(landmarks[j]) * width)*scale_x) + x1
            y = ((y_center - float(landmarks[j + 1]) * height)*scale_y) + y1

            landmark_arr_xy.append(x)
            landmark_arr_xy.append(y)

            landmark_arr_x.append(x)
            landmark_arr_y.append(y)

            img = cv2.circle(img, (int(x), int(y)), 2, (255, 14, 74), 2)
            img = cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), 1)

        return landmark_arr_xy, landmark_arr_x, landmark_arr_y, img


    def create_landmarks_from_normalized(self, landmarks, width, height, x_center, y_center):

        # landmarks_splited = _landmarks.split(';')
        landmark_arr_xy = []
        landmark_arr_x = []
        landmark_arr_y = []

        for j in range(0, len(landmarks), 2):
            x = x_center - float(landmarks[j]) * width
            y = y_center - float(landmarks[j + 1]) * height

            landmark_arr_xy.append(x)
            landmark_arr_xy.append(y)  # [ x1, y1, x2,y2 ]

            landmark_arr_x.append(x)  # [x1, x2]
            landmark_arr_y.append(y)  # [y1, y2]

        return landmark_arr_xy, landmark_arr_x, landmark_arr_y

    def create_landmarks(self, landmarks, scale_factor_x, scale_factor_y):
        # landmarks_splited = _landmarks.split(';')
        landmark_arr_xy = []
        landmark_arr_x = []
        landmark_arr_y = []
        for j in range(0, len(landmarks), 2):

            x = float(landmarks[j]) * scale_factor_x
            y = float(landmarks[j + 1]) * scale_factor_y

            landmark_arr_xy.append(x)
            landmark_arr_xy.append(y)  # [ x1, y1, x2,y2 ]

            landmark_arr_x.append(x)  # [x1, x2]
            landmark_arr_y.append(y)  # [y1, y2]

        return landmark_arr_xy, landmark_arr_x, landmark_arr_y

    def create_landmarks_aflw(self, landmarks, scale_factor_x, scale_factor_y):
        # landmarks_splited = _landmarks.split(';')
        landmark_arr_xy = []
        landmark_arr_x = []
        landmark_arr_y = []
        for j in range(0, len(landmarks), 2):
            if landmarks[j][0] == 1:
                x = float(landmarks[j][1]) * scale_factor_x
                y = float(landmarks[j][2]) * scale_factor_y

                landmark_arr_xy.append(x)
                landmark_arr_xy.append(y)  # [ x1, y1, x2,y2 ]

                landmark_arr_x.append(x)  # [x1, x2]
                landmark_arr_y.append(y)  # [y1, y2]

        return landmark_arr_xy, landmark_arr_x, landmark_arr_y

    def random_augmentation(self, lbl, img, number_of_landmark):
        # a = random.randint(0, 2)
        # if a == 0:
        #     img, lbl = self.__add_margin(img, img.shape[0], lbl)

        '''this function has problem!!!'''
        # img, lbl = self.__add_margin(img, img.shape[0], lbl)

        # else:
        #     img, lbl = self.__negative_crop(img, lbl)

        # i = random.randint(0, 2)
        # if i == 0:
        #     img, lbl = self.__rotate(img, lbl, 90, img.shape[0], img.shape[1])
        # elif i == 1:
        #     img, lbl = self.__rotate(img, lbl, 180, img.shape[0], img.shape[1])
        # else:
        #     img, lbl = self.__rotate(img, lbl, 270, img.shape[0], img.shape[1])

        # k = random.randint(0, 3)
        # if k > 0:
        #     img = self.__change_color(img)
        #
        img = self.__noisy(img)

        lbl = np.reshape(lbl, [number_of_landmark*2])
        return lbl, img


    def cropImg_2time(self, img, x_s, y_s):
        min_x = max(0, int(min(x_s) - 100))
        max_x = int(max(x_s) + 100)
        min_y = max(0, int(min(y_s) - 100))
        max_y = int(max(y_s) + 100)

        crop = img[min_y:max_y, min_x:max_x]

        new_x_s = []
        new_y_s = []
        new_xy_s = []

        for i in range(len(x_s)):
            new_x_s.append(x_s[i] - min_x)
            new_y_s.append(y_s[i] - min_y)
            new_xy_s.append(x_s[i] - min_x)
            new_xy_s.append(y_s[i] - min_y)
        return crop, new_xy_s

    def cropImg(self, img, x_s, y_s, no_padding=False):
        margin1 = random.randint(0, 10)
        margin2 = random.randint(0, 10)
        margin3 = random.randint(0, 10)
        margin4 = random.randint(0, 10)

        if no_padding:
            min_x = max(0, int(min(x_s)))
            max_x = int(max(x_s))
            min_y = max(0, int(min(y_s)))
            max_y = int(max(y_s))
        else:
            min_x = max(0, int(min(x_s) - margin1))
            max_x = int(max(x_s) + margin2)
            min_y = max(0, int(min(y_s) - margin3))
            max_y = int(max(y_s) + margin4)

        crop = img[min_y:max_y, min_x:max_x]

        new_x_s = []
        new_y_s = []
        new_xy_s = []

        for i in range(len(x_s)):
            new_x_s.append(x_s[i] - min_x)
            new_y_s.append(y_s[i] - min_y)
            new_xy_s.append(x_s[i] - min_x)
            new_xy_s.append(y_s[i] - min_y)

        # imgpr.print_image_arr(k, crop, new_x_s, new_y_s)
        # imgpr.print_image_arr_2(i, img, x_s, y_s, [min_x, max_x], [min_y, max_y])

        return crop, new_xy_s

    def __negative_crop(self, img, landmarks):

        landmark_arr_xy, x_s, y_s = self.create_landmarks(landmarks, 1, 1)
        min_x = img.shape[0] // random.randint(5, 15)
        max_x = img.shape[0] - (img.shape[0] // random.randint(15, 20))
        min_y = img.shape[0] // random.randint(5, 15)
        max_y = img.shape[0] - (img.shape[0] // random.randint(15, 20))

        crop = img[min_y:max_y, min_x:max_x]

        new_x_s = []
        new_y_s = []
        new_xy_s = []

        for i in range(len(x_s)):
            new_x_s.append(x_s[i] - min_x)
            new_y_s.append(y_s[i] - min_y)
            new_xy_s.append(x_s[i] - min_x)
            new_xy_s.append(y_s[i] - min_y)

        # imgpr.print_image_arr(crop.shape[0], crop, new_x_s, new_y_s)
        # imgpr.print_image_arr_2(crop.shape[0], crop, x_s, y_s, [min_x, max_x], [min_y, max_y])

        return crop, new_xy_s

    def __add_margin(self, img, img_w, lbl):
        marging_width = img_w // random.randint(15, 20)
        direction = random.randint(0, 4)

        if direction == 1:
            margings = np.random.random([img_w, int(marging_width), 3])
            img = np.concatenate((img, margings), axis=1)

        if direction == 2:
            margings_1 = np.random.random([img_w, int(marging_width), 3])
            img = np.concatenate((img, margings_1), axis=1)

            marging_width_1 = img_w // random.randint(15, 20)
            margings_2 = np.random.random([int(marging_width_1), img_w + int(marging_width), 3])
            img = np.concatenate((img, margings_2), axis=0)

        if direction == 3:  # need chane labels
            margings_1 = np.random.random([img_w, int(marging_width), 3])
            img = np.concatenate((margings_1, img), axis=1)
            lbl = self.__transfer_lbl(int(marging_width), lbl, [1, 0])

            marging_width_1 = img_w // random.randint(15, 20)
            margings_2 = np.random.random([int(marging_width_1), img_w + int(marging_width), 3])
            img = np.concatenate((margings_2, img), axis=0)
            lbl = self.__transfer_lbl(int(marging_width_1), lbl, [0, 1])

        if direction == 4:  # need chane labels
            margings_1 = np.random.random([img_w, int(marging_width), 3])
            img = np.concatenate((margings_1, img), axis=1)
            lbl = self.__transfer_lbl(int(marging_width), lbl, [1, 0])
            img_w1 = img_w + int(marging_width)

            marging_width_1 = img_w // random.randint(15, 20)
            margings_2 = np.random.random([int(marging_width_1), img_w1, 3])
            img = np.concatenate((margings_2, img), axis=0)
            lbl = self.__transfer_lbl(int(marging_width_1), lbl, [0, 1])
            img_w2 = img_w + int(marging_width_1)

            marging_width_1 = img_w // random.randint(15, 20)
            margings_1 = np.random.random([img_w2, int(marging_width_1), 3])
            img = np.concatenate((img, margings_1), axis=1)

            marging_width_1 = img_w // random.randint(15, 20)
            margings_2 = np.random.random([int(marging_width_1), img.shape[1], 3])
            img = np.concatenate((img, margings_2), axis=0)

        return img, lbl

    def __void_image(self, img, img_w, ):
        marging_width = int(img_w / random.randint(7, 16))
        direction = random.randint(0, 1)
        direction = 0
        if direction == 0:
            np.delete(img, 100, 1)
            # img[:, 0:marging_width, :] = 0
        elif direction == 1:
            img[img_w - marging_width:img_w, :, :] = 0
        if direction == 2:
            img[:, img_w - marging_width:img_w, :] = 0

        return img

    def __change_color(self, img):
        # color_arr = np.random.random([img.shape[0], img.shape[1]])
        color_arr = np.zeros([img.shape[0], img.shape[1]])
        axis = random.randint(0, 4)

        if axis == 0:  # red
            img_mono = img[:, :, 0]
            new_img = np.stack([img_mono, color_arr, color_arr], axis=2)
        elif axis == 1:  # green
            img_mono = img[:, :, 1]
            new_img = np.stack([color_arr, img_mono, color_arr], axis=2)
        elif axis == 2:  # blue
            img_mono = img[:, :, 1]
            new_img = np.stack([color_arr, img_mono, color_arr], axis=2)
        elif axis == 3:  # gray scale
            img_mono = img[:, :, 0]
            new_img = np.stack([img_mono, img_mono, img_mono], axis=2)
        else:  # random noise
            color_arr = np.random.random([img.shape[0], img.shape[1]])
            img_mono = img[:, :, 0]
            new_img = np.stack([img_mono, img_mono, color_arr], axis=2)

        return new_img

    def __rotate_origin_only(self, xy_arr, radians, xs, ys):
        """Only rotate a point around the origin (0, 0)."""
        rotated = []
        for xy in xy_arr:
            x, y = xy
            xx = x * math.cos(radians) + y * math.sin(radians)
            yy = -x * math.sin(radians) + y * math.cos(radians)
            rotated.append([xx + xs, yy + ys])
        return np.array(rotated)

    def __rotate(self, img, landmark_old, degree, img_w, img_h, num_of_landmarks):
        landmark_old = np.reshape(landmark_old, [num_of_landmarks, 2])

        theta = math.radians(degree)

        if degree == 90:
            landmark = self.__rotate_origin_only(landmark_old, theta, 0, img_h)
            return np.rot90(img, 3, axes=(-2, 0)), landmark
        elif degree == 180:
            landmark = self.__rotate_origin_only(landmark_old, theta, img_h, img_w)
            return np.rot90(img, 2, axes=(-2, 0)), landmark
        elif degree == 270:
            landmark = self.__rotate_origin_only(landmark_old, theta, img_w, 0)
            return np.rot90(img, 1, axes=(-2, 0)), landmark

    def __transfer_lbl(self, marging_width_1, lbl, axis_arr):
        new_lbl = []
        for i in range(0, len(lbl), 2):
            new_lbl.append(lbl[i] + marging_width_1 * axis_arr[0])
            new_lbl.append(lbl[i + 1] + marging_width_1 * axis_arr[1])
        return np.array(new_lbl)