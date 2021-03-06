import os
import numpy as np
import PIL.Image as pilimg
from PIL import ImageOps
#import scipy.ndimage
# 내부 모듈
from data_model import DataModel
from threaded_generator import threadsOn
import multiprocessing

# img, label을 따로 할때 순서와 동기화 문제로 한 세트당 처리하는게 이득임

def test(data_model: DataModel):
    file_name_sets = find_file_name_sets(data_model)
    pre_sets = open_pre_sets(data_model=data_model, file_name_sets= file_name_sets)

    with multiprocessing.Pool(8) as pool:
        pre_set_args = create_rotated_set_args(data_model, pre_sets)
        rotated_tables = pool.map(create_rotated_set, pre_set_args)
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
                print(f"find data x:{input_file_name}, y:{output_file_name}")
                yield (input_file_name, output_file_name)

def open_pre_sets(data_model: DataModel, file_name_sets):
    for file_name_set in file_name_sets:
        input_file_name, output_file_name = file_name_set

        pre_input = _open_pre_input(
            data_model=data_model, input_file_name=input_file_name)
        pre_output = _open_pre_output(
            data_model=data_model, output_file_name=output_file_name)

        yield (pre_input, pre_output)

def create_rotated_set_args(data_model: DataModel, pre_sets):
    for pre_set in pre_sets:
        pre_input, pre_output = pre_set
        pre_input: pilimg.Image
        pre_output: pilimg.Image
        for angle in range(360):
            yield (data_model, pre_input, pre_output, angle)

def create_rotated_set(set_arg):
    data_model, pre_input, pre_output, angle = set_arg
    pre_input: pilimg.Image
    pre_output: pilimg.Image
    rotated_output = np.array(pre_output.rotate(angle).resize(data_model.output_shape[0:2]))
    rotated_output = _threshold(rotated_output) * 255
    return (np.array(pre_input.rotate(angle).resize(data_model.input_shape[0:2])), rotated_output)

def create_rotated_sets(data_model: DataModel, pre_set):
    pre_input, pre_output = pre_set
    pre_input: pilimg.Image
    pre_output: pilimg.Image

    for angle in range(360):
        yield (pre_input.rotate(angle).resize(data_model.input_shape[0:2]), pre_output.rotate(angle).resize(data_model.output_shape[0:2]))

def _open_pre_input(data_model: DataModel, input_file_name):
    raw_input = pilimg.open(f"{data_model.img_path}/{input_file_name}")
    raw_input = ImageOps.exif_transpose(raw_input)
    return raw_input.resize(data_model.pre_shape[0:2])

def _open_pre_output(data_model: DataModel, output_file_name):
    read_lines = _open_and_read_lines(
        f"{data_model.label_path}/{output_file_name}")
    numeric_lines = _convert_numeric_line(read_lines)
    left_top_numeric_lines = _center_to_left_top_lines(numeric_lines)
    scaled_numberic_lines = _scale_numeric_lines(
        data_model=data_model, lines=left_top_numeric_lines)
    merged_lines = _merge_lines(
        data_model, scaled_numberic_lines)
    convolved_imgs = _apply_avg_filter(merged_lines)
    return pilimg.fromarray((convolved_imgs * 255).astype(np.uint8))

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
        yield (label, x, y, w, h)


def _merge_lines(data_model: DataModel, lines):
    pre_shape = list(data_model.pre_shape)
    pre_shape[-1] = data_model.output_shape[-1]
    result = np.zeros(pre_shape)
    for line in lines:
        label, x, y, w, h = line
        for pos_x in range(w):
            for pos_y in range(h):
                result[pos_y + y, pos_x + x, label] = 1.
    return result

avg_filter = np.ones([3,3]) / 3 / 3
def _threshold(x):
    if x > 0.01:
        return 1.0
    else:
        return 0.0
_threshold = np.vectorize(_threshold)



def _apply_avg_filter(merged_lines: np.ndarray):
    convolved_img = merged_lines.copy()
    #convolved_img[:,:,0] = scipy.ndimage.convolve(merged_lines[:,:,0], avg_filter, mode="constant")
    #convolved_img[:,:,1] = scipy.ndimage.convolve(merged_lines[:,:,1], avg_filter, mode="constant")
    return _threshold(convolved_img)



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


if __name__ == '__main__':
    test_data_model = DataModel("./data/x", "./data/y", (256,256,3),(256,256,3), (50, 50, 2))
    test(test_data_model)