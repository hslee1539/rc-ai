
class DataModel:
    img_path : str
    label_path : str
    training_input_path : str = "train-data/input"
    training_output_path : str = "train-data/output"
    pre_shape : tuple
    input_shape : tuple
    output_shape : tuple

    def __init__(self, img_path: str, label_path: str, pre_shape: tuple, input_shape: tuple, output_shape: tuple):
        """"""
        self.img_path = img_path
        self.label_path = label_path
        self.pre_shape = pre_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
