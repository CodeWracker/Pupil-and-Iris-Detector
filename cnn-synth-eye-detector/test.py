import os
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
def test():
    paths = (os.listdir("./models"))
    print("\n\nSaved Models:")
    for i,path in enumerate(paths):
        print(str(i) + " - " + str(path))
    ch = int(input("Choose a model to test: "))
    model = keras.models.load_model("./models/" + paths[ch])

    img = cv2.imread("./img/"+input("write the path to the image: img/"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Test Image",img)
    cv2.waitKey(0)
    plt.imshow(img,cmap="gray")
    plt.show()

    imgT = img.reshape(1,80,120,1)
    print(imgT.shape)
    predictions = model.predict(imgT)
    predictions = predictions[0]
    
    
    x_eye = int(math.sqrt((predictions[0] )**2))
    y_eye = int(math.sqrt((predictions[1])**2))
    rd_iris = int(math.sqrt((predictions[2] )**2))
    rd_pupil = int(math.sqrt((predictions[3] )**2))

    '''x_eye = int(math.sqrt((predictions[0] * 120 )**2))
    y_eye = int(math.sqrt((predictions[1] * 80 )**2))
    rd_iris = int(math.sqrt((predictions[2]  * 120)**2))
    rd_pupil = int(math.sqrt((predictions[3] * 120 )**2))'''

    
    print("Results")
    print(x_eye,y_eye,rd_iris,rd_pupil)
    img = cv2.circle(img,(x_eye,y_eye),int(rd_pupil),(255, 255, 255),1)
    img = cv2.circle(img,(x_eye,y_eye),int(rd_iris),(255, 255, 255),1)
    cv2.imshow("Img",img)
    
    cv2.waitKey(0)
if __name__ == "__main__":
    test()