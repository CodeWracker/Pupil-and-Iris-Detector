import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
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
from tqdm import tqdm

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
def train():
    print(tf.__version__)

    NAME = f'eye-cambrige-80x120-cnn-points-{int(time.time())}'
    


    file = open('./processed/results.pickle', 'rb')
    data = pickle.load(file)
    images = np.array(data['image'])
    #print(images.shape)


    iris_points,pupil_points = data['iris_points'],data['pupil_points']
    Y = []
    for i in tqdm(range(0,len(iris_points))):
        aux = []
        for pt in iris_points[i]:
            x,y = pt
            aux.append(x/120)
            aux.append(y/80)
        for pt in pupil_points[i]:
            x,y = pt
            aux.append(x/120)
            aux.append(y/80)
        Y.append(aux)
    Y = np.array(Y)
    ''' antigos outputs
    X,Y,Ri,Rp = np.array(data['x_eye']),np.array(data['y_eye']),np.array(data['iris_radius']),np.array(data['pupil_radius'])
    #X = X/120
    #Y = Y/80
    #Ri = Ri/120
    #Rp = Rp/120
    y = []
    for i in range(0,len(X)):
        y.append(np.array([X[i],Y[i],Ri[i],Rp[i]]))
    y = np.array(y)'''




    x_train,x_test,y_train,y_test = train_test_split(images,Y)
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

    model.add(tf.keras.layers.Convolution2D(50,(3,3),input_shape=x_train[0].shape))
    model.add(tf.keras.layers.Activation('relu'))


    model.add(tf.keras.layers.Convolution2D(24,(3,3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.05))


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.Dropout(0.05))

    model.add(tf.keras.layers.Dense(nb_classes))
    model.add(tf.keras.layers.Activation('linear'))
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    print(model.summary())

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

    model.fit(x = x_train,
                y =  y_train,
                epochs =50,
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

    converter = [120,80]
    for i in range(len(predictions)):
        predictions[i] = int(math.sqrt((predictions[i] * converter[i%2] )**2))
    
    img = x_test[67].reshape(80,120)
    iris_points,pupil_points = parse_predictions(predictions)
    for pt in iris_points:
        x,y = pt
        cv2.putText(img,'.',(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,0.8,(255, 255, 255),1)
    for pt in pupil_points:
        x,y = pt
        cv2.putText(img,'.',(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,0.8,(255, 255, 255),1)



    print("Results")
    print(iris_points,pupil_points)
    '''print(x_eye,y_eye,rd_iris,rd_pupil)
    img = cv2.circle(img,(x_eye,y_eye),int(rd_pupil),(255, 255, 255),1)
    img = cv2.circle(img,(x_eye,y_eye),int(rd_iris),(255, 255, 255),1)'''
    cv2.imshow("Img",img)
    
    cv2.waitKey(0)
    
    plt.imshow(img,cmap='gray')
    plt.show()
if __name__ == "__main__":
    train()