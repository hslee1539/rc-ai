# 모듈
import os
import numpy as np
import PIL.Image as pilimg
import matplotlib.pyplot as plt

# 내부 모듈

from data_function import *
from data_model import DataModel


def _end(args):
    return args[1]

def load_img(data_model: DataModel):
    img_files = start_img(data_model)
    raw_imgs = map(file_to_img, img_files)
    resized_imgs = map(resize_input_img, raw_imgs)
    imgs = list(map(_end, resized_imgs))
    return np.array(imgs)

def load_rotated_imgs(data_model: DataModel):
    img_files = start_img(data_model)
    raw_imgs = map(file_to_img, img_files)
    rotated_imgs = map(create_rotated_imgs, raw_imgs)
    resized_imgs = resize_rotated_imgs(rotated_imgs)
    imgs = list(resized_imgs)
    return np.array(imgs)



    

def load_label_img(data_model: DataModel):
    label_files = start_label(data_model)
    raw_labels = map(file_to_raw_label, label_files)
    number_labels = map(raw_to_number_label, raw_labels)
    number_labels = map(center_to_left_top_label, number_labels)
    scaled_labels = map(scale_to_output, number_labels)
    merged_label_imgs = merge_number_to_label_util_none(scaled_labels)
    label_imgs = list(map(_end, merged_label_imgs))
    return np.array(label_imgs)





def load():
    """"""
    for fileName in os.listdir(PATH_IMGS + "/" + path):
        """"""


def _resize(src, w, h):
    shape = list(src.shape)
    old_h = shape[1]
    old_w = shape[2]
    shape[1] = h
    shape[2] = w

    result = np.zeros(shape)

    for i in range(shape[0]):
        for y in range(shape[1]):
            a1 = y * old_h // shape[1]
            for x in range(shape[2]):
                a2 = x * old_w // shape[2]
                for c in range(shape[3]):
                    result[i, y, x, c] = src[i, a1, a2, c]
    return result


def _apply_mask(src, mask, w, h):
    resized_src = _resize(src, w, h)
    resized_src *= mask
    return resized_src


def rotate_test():
    test_data_model = DataModel("./data/x", "./data/y", (256,256,3), (50, 50, 2))

    img_files = start_img(test_data_model)
    raw_imgs = map(file_to_img, img_files)
    for img in raw_imgs:
        target = img[1]
        target: pilimg.Image
        target = target.resize((100,100))
        plt.imshow(target)
        plt.show()
        for item in rotate_imgs(np.array(target)):
            plt.imshow(item)
            plt.show()
        img_ = _resize(np.array([np.array(img[1])]), 256,256)
        img_ = np.array(img_[0] , dtype=np.uint8)
        pilimg.fromarray(img_).save("test.jpg")
        plt.imshow()
        plt.show()
        break


if __name__ == "__main__":
    """"""
    test_data_model = DataModel("./data/x", "./data/y", (256,256,3), (50, 50, 2))

    x1 = load_rotated_imgs(test_data_model)
    y = load_label_img(test_data_model)
    x = load_img(test_data_model)

    for i in range(30):
        resized = _resize(x[i: i + 1], test_data_model.output_shape[1], test_data_model.output_shape[0])

        mask = _apply_mask(x[i: i + 1], y[i: i + 1, :, :, 0: 1], test_data_model.output_shape[1], test_data_model.output_shape[0])
        plt.imshow((mask[0] * 250).astype(np.uint8))
        plt.show()
        #plt.imshow(resized[0])
        #plt.show()
        plt.imshow(y[i, : , : , 0], vmin=0., vmax=1.)
        plt.show()
        plt.imshow(x[i], vmin=0., vmax = 1.)
        plt.show()
    print(y)
