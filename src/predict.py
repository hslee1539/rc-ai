
import data
import network
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import PIL.Image as pilimg
import cv2

## 맥북 실행법 -> 터미널에서 python3 predict.py 직접 실행

net = network.load("my-net-v1")

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def do_thr(t):
    if (t > 0.005):
        return 1.0
    else:
        return 0.0

do_thr = np.vectorize(do_thr)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    if frame is None:
        continue
    origen = np.copy(frame)
    frame = pilimg.fromarray(frame)
    frame = frame.resize((100,100))
    frame = np.array(frame)
    predict = net.predict(frame.reshape((1,100,100,3)))[0]
    predict = np.insert(predict, 0, 0, -1)
    predict = do_thr(predict)
    predict *= 250
    predict = predict.astype(np.uint8)
    predict = pilimg.fromarray(predict)
    predict = predict.resize((480,640))

    output = np.insert(origen, 1, predict, 1)
    
    
    cv2.imshow("VideoFrame", output)

capture.release()
cv2.destroyAllWindows()

