import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
def validate():
    paths = (os.listdir("./models"))
    print("\n\nSaved Models:")
    for i,path in enumerate(paths):
        print(str(i) + " - " + str(path))
    ch = int(input("Choose a model to validate: "))
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    model = keras.models.load_model("./models/" + paths[ch])
    
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    results = model.evaluate(x_test, y_test, batch_size=10)
    print("test loss, test acc:", results)