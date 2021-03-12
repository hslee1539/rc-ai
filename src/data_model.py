
class DataModel:
    img_path : str
    label_path : str
    input_shape : tuple
    output_shape : tuple

    def __init__(self, img_path: str, label_path: str, input_shape, output_shape):
        """"""
        self.img_path = img_path
        self.label_path = label_path
        self.input_shape = input_shape
        self.output_shape = output_shape
