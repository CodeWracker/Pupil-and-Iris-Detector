import os
from numpy.core.fromnumeric import argmax
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
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
    img = cv2.bitwise_not(img)
    plt.imshow(img,cmap=plt.cm.binary)
    plt.show()

    img = img.reshape(1,28,28,1)
    
    res = model.predict(img)
    print(np.argmax(res))
