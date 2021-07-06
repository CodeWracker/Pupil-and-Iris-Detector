import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
def train():
  print(tf.__version__)

  NAME = f'mnist-28x28-cnn-{int(time.time())}'
  

  mnist = tf.keras.datasets.mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()


  plt.imshow(x_train[0],cmap=plt.cm.binary)
  plt.show()



  # como eu coloquei cnn, eu fiz um reshape pra ter 3 dim
  print(x_train.shape)
  print(x_test.shape)
  x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
  x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
  print()
  print(x_train.shape)
  print(x_test.shape)

  # aqui eu dei uma modificada do cod la em baixo pra usar cnn

  nb_classes=10
  nb_filters=32
  nb_conv=3

  model= tf.keras.models.Sequential()

  model.add(tf.keras.layers.Convolution2D(nb_filters,(nb_conv,nb_conv),padding="same",input_shape=(28,28,1)))
  model.add(tf.keras.layers.Activation('relu'))


  model.add(tf.keras.layers.Convolution2D(nb_filters,(nb_conv,nb_conv)))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dropout(0.2))


  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(24))
  model.add(tf.keras.layers.Dropout(0.2))


  model.add(tf.keras.layers.Dense(nb_classes))
  model.add(tf.keras.layers.Activation('softmax'))
  model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  print(model.summary())

  #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

  model.fit(x_train, y_train,
            epochs=5)

  print(x_test[67].shape)
  plt.imshow(x_test[67].reshape(28,28),cmap=plt.cm.binary)
  plt.show()
  predictions = model.predict(x_test[67].reshape(1,28,28,1))
  print(np.argmax(predictions))

  model.save('models/'+NAME)
  