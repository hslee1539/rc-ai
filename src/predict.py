import data
import network
import matplotlib.pyplot as plt
import numpy as np
import cv2
def predict_camera():
    cv2.namedWindow("VideoFrame")
    model = network.load("my-net-v1")
    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_camera()
