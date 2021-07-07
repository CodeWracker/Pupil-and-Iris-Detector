import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import cv2
from tensorflow.keras.callbacks import TensorBoard
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
from sklearn.model_selection import train_test_split  
import math


def train():
    print(tf.__version__)

    NAME = f'eye-cambrige-80x120-cnn-{int(time.time())}'
    


    file = open('./processed/results.pickle', 'rb')
    data = pickle.load(file)
    images = np.array(data['image'])
    #print(images.shape)
    X,Y,Ri,Rp = np.array(data['x_eye']),np.array(data['y_eye']),np.array(data['iris_radius']),np.array(data['pupil_radius'])
    #X = X/120
    #Y = Y/80
    #Ri = Ri/120
    #Rp = Rp/120
    y = []
    for i in range(0,len(X)):
        y.append(np.array([X[i],Y[i],Ri[i],Rp[i]]))
    y = np.array(y)
    x_train,x_test,y_train,y_test = train_test_split(images,y)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    


    plt.imshow(x_test[0],cmap='gray')
    plt.show()


    
    

    # aqui eu dei uma modificada do cod la em baixo pra usar cnn

    nb_classes=y_train[0].shape[0]
    nb_filters=32
    nb_conv=3

    model= tf.keras.models.Sequential()

    model.add(tf.keras.layers.Convolution2D(32,(3,3),input_shape=x_train[0].shape))
    model.add(tf.keras.layers.Activation('relu'))


    model.add(tf.keras.layers.Convolution2D(32,(3,3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.2))


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(24))
    model.add(tf.keras.layers.Dropout(0.2))


    model.add(tf.keras.layers.Dense(nb_classes))
    model.add(tf.keras.layers.Activation('linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    print(model.summary())

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

    model.fit(x = x_train,
                y =  y_train,
                epochs =10,
               validation_data=(x_test,y_test),
               callbacks = [tensorboard] )
    
    model.save('models/'+NAME)
    print(x_test[67].shape)
    plt.imshow(x_test[67].reshape(80,120),cmap='gray')
    plt.show()
    aux_x = np.array([x_test[67]])
    print(aux_x.shape)
    predictions = model.predict(aux_x)
    print(predictions)
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
    img = x_test[67].reshape(80,120)
    img = cv2.circle(img,(x_eye,y_eye),int(rd_pupil),(255, 255, 255),1)
    img = cv2.circle(img,(x_eye,y_eye),int(rd_iris),(255, 255, 255),1)
    cv2.imshow("Img",img)
    
    cv2.waitKey(0)

if __name__ == "__main__":
    train()