import os
import numpy as np
import PIL.Image as pilimg
import scipy.ndimage
# 내부 모듈
from data_model import DataModel
from threaded_generator import threadsOn

# img, label을 따로 할때 순서와 동기화 문제로 한 세트당 처리하는게 이득임


def test(data_model: DataModel):
    file_name_sets = find_file_name_sets(data_model)
    pre_sets = open_pre_sets(data_model=data_model, file_name_sets= file_name_sets)
    rotated_tables = map(create_rotated_sets, pre_sets)
    rotated_tables = threadsOn(rotated_tables) # pre_sets 수 만큼 쓰래드 생성



def find_file_name_sets(data_model: DataModel):
    input_file_names = os.listdir(data_model.img_path)
    output_file_names = os.listdir(data_model.label_path)

    for input_file_name in input_file_names:
        if not input_file_name.endswith(".jpg"):
            continue

        output_file_name = None
        for output_file_name in output_file_names:
            if output_file_name.replace(".txt", "") == input_file_name.replace(".jpg", ""):
                yield (input_file_name, output_file_name)


def open_pre_sets(data_model: DataModel, file_name_sets):
    for file_name_set in file_name_sets:
        input_file_name, output_file_name = file_name_set

        pre_input = _open_pre_input(
            data_model=data_model, input_file_name=input_file_name)
        pre_output = _open_pre_output(
            data_model=data_model, output_file_name=output_file_name)

        yield (pre_input, pre_output)


def create_rotated_sets(data_model: DataModel, pre_set):
    pre_input, pre_output = pre_set
    pre_input: pilimg.Image
    pre_output: pilimg.Image

    for angle in range(360):
        yield (pre_input.rotate(angle).resize(data_model.input_shape[0:2]), pre_output.rotate(angle).resize(data_model.output_shape[0:2]))




def _open_pre_input(data_model: DataModel, input_file_name):
    raw_input = pilimg.open(f"{data_model.img_path}/{input_file_name}")
    return raw_input.resize(data_model.pre_shape[0:2])


def _open_pre_output(data_model: DataModel, output_file_name):
    read_lines = _open_and_read_lines(
        f"{data_model.label_path}/{output_file_name}")
    numeric_lines = _convert_numeric_line(read_lines)
    left_top_numeric_lines = _center_to_left_top_lines(numeric_lines)
    scaled_numberic_lines = _scale_numeric_lines(
        data_model=data_model, lines=left_top_numeric_lines)
    output_ndimage = _merge_and_convert_ndimage(
        data_model, scaled_numberic_lines)
    return pilimg.fromarray(output_ndimage)


def _open_and_read_lines(file_name):
    with open(f"{file_name}", "r") as file:
        retval = file.readline()
        while len(retval) != 0:
            yield retval.replace("\n", "").split(" ")
            retval = file.readline()


def _convert_numeric_line(lines):
    for line in lines:
        label, x, y, w, h = line
        yield (int(label), float(x), float(y), float(w), float(h))


def _scale_numeric_lines(data_model: DataModel, lines):
    for line in lines:
        label, x, y, w, h = line
        x = int(x * data_model.pre_shape[1])
        y = int(y * data_model.pre_shape[0])
        w = int(w * data_model.pre_shape[1])
        h = int(h * data_model.pre_shape[0])


def _merge_and_convert_ndimage(data_model: DataModel, lines):
    result = np.zeros(data_model.pre_shape)
    for line in lines:
        label, x, y, w, h = line
        for pos_x in range(w):
            for pos_y in range(h):
                result[pos_y + y, pos_x + x, label] = 1.
    return result


def start_img(data_model: DataModel):
    files = os.listdir(data_model.img_path)
    files.sort()
    for file_name in files:
        if not file_name.endswith(".jpg"):
            continue
        yield (data_model, file_name)


def file_to_img(args):
    data_model, file_name = args
    data_model: DataModel
    img = pilimg.open(f"{data_model.img_path}/{file_name}")
    return (data_model, img)


def create_rotated_imgs(args):
    data_model, img = args
    data_model: DataModel
    if img.__class__ is not np.ndarray:
        img = np.array(img)

    for angle in range(360):
        rotated_img = scipy.ndimage.rotate(img, angle)
        yield (data_model, rotated_img)


def resize_rotated_imgs(datas):
    i = 0
    for data in datas:
        for args in data:
            print(f"- {i}")
            i += 1
            data_model, img = args
            if img.__class__ is not pilimg.Image:
                img = pilimg.fromarray(img)

            resized_img = img.resize(data_model.input_shape[0:2])

            yield (data_model, resized_img)


def resize_input_img(args):
    data_model, img = args
    data_model: DataModel
    img: pilimg.Image
    resized_img = np.array(img.resize(data_model.input_shape[0:2]))
    return (data_model, resized_img)


def start_label(data_model: DataModel):
    files = os.listdir(data_model.label_path)
    files.sort()
    for file_name in files:
        file_name: str
        if not file_name.endswith(".txt") or file_name.endswith("classes.txt"):
            continue
        yield (data_model, file_name)


def file_to_raw_label(args):
    data_model, file_name = args
    data_model: DataModel

    with open(f"{data_model.label_path}/{file_name}", "r") as file:
        retval = file.readline()
        while len(retval) != 0:
            yield (data_model, retval.replace("\n", "").split(" "))
            retval = file.readline()


def raw_to_number_label(data):
    for args in data:
        data_model, raw_label = args
        data_model: DataModel
        label, center_x, center_y, w, h = raw_label
        yield (data_model, (int(label), float(center_x), float(center_y), float(w), float(h)))


def _center_to_left_top_lines(lines):
    for line in lines:
        label, center_x, center_y, w, h = line

        x = center_x
        if center_x - w / 2 > 0:
            x -= w / 2
        y = center_y
        if center_y - h / 2 > 0:
            y -= h / 2

        yield (label, x, y, w, h)


def scale_to_output(data):
    for args in data:
        data_model, number_label = args
        data_model: DataModel

        label, x, y, w, h = number_label

        x = int(x * data_model.output_shape[1])
        y = int(y * data_model.output_shape[0])
        w = int(w * data_model.output_shape[1])
        h = int(h * data_model.output_shape[0])

        yield (data_model, (label, x, y, w, h))


def merge_number_to_label_util_none(datas):
    result = None
    data_model = None
    for data in datas:
        is_first = True
        for args in data:
            data_model, number_label = args
            data_model: DataModel

            if is_first:
                result = np.zeros(data_model.output_shape)
                is_first = False

            label, x, y, w, h = number_label

            for pos_x in range(w):
                for pos_y in range(h):
                    result[pos_y + y, pos_x + x, label] = 1.
        yield (data_model, result)


def resize_img(img, w, h):
    shape = list(img.shape)
    old_h = shape[0]
    old_w = shape[1]
    shape[0] = h
    shape[1] = w

    result = np.zeros(shape, dtype=img.dtype)

    for y in range(shape[0]):
        a1 = y * old_h // shape[0]
        for x in range(shape[1]):
            a2 = x * old_w // shape[1]
            for c in range(shape[2]):
                result[y, x, c] = img[a1, a2, c]
    return result


def rotate_imgs(img):
    ""
    shape = img.shape
    for i in range(360):
        ""
        rotated_img = scipy.ndimage.rotate(img, i)
        yield resize_img(rotated_img, shape[0], shape[1])
