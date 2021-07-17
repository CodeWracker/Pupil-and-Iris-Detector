import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from numpy.core.fromnumeric import argmax
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
def parse_predictions(preds):
    iris_points = [
        [preds[0],preds[1]],
        [preds[2],preds[3]],
        [preds[4],preds[5]],
        [preds[6],preds[7]],
        [preds[8],preds[9]],
        [preds[10],preds[11]],
        [preds[12],preds[13]],
        [preds[14],preds[15]],
    ]
    pupil_points = [
        [preds[16],preds[17]],
        [preds[18],preds[19]],
        [preds[20],preds[21]],
        [preds[22],preds[23]],
        [preds[24],preds[25]],
        [preds[26],preds[27]],
        [preds[28],preds[29]],
        [preds[30],preds[31]],
    ]
    return iris_points,pupil_points
def test():
    paths = (os.listdir("./models"))
    print("\n\nSaved Models:")
    for i,path in enumerate(paths):
        print(str(i) + " - " + str(path))
    ch = int(input("Choose a model to test: "))
    model = keras.models.load_model("./models/" + paths[ch])

    img = cv2.imread("./img/"+input("write the path to the image: img/"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(img,cmap="gray")
    plt.show()

    imgT = img.reshape(1,80,120,1)
    print(imgT.shape)
    predictions = model.predict(imgT)
    predictions = predictions[0]
    converter = [120,80]
    for i in range(len(predictions)):
        predictions[i] = int(math.sqrt((predictions[i] * converter[i%2] )**2))
    
    iris_points,pupil_points = parse_predictions(predictions)
    for pt in iris_points:
        x,y = pt
        cv2.putText(img,'.',(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,0.8,(255, 255, 255),1)
    for pt in pupil_points:
        x,y = pt
        cv2.putText(img,'.',(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,0.8,(255, 255, 255),1)
    plt.imshow(img,cmap="gray")
    plt.show()
if __name__ == "__main__":
    test()