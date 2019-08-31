from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from make_tensorboard import make_tensorboard

np.random.seed(1671)
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT=0.2
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("{0}, {1}, {2}, {3}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

np.set_printoptions(linewidth=150)
print("{0}".format(x_train[0]))
print("{0}".format(y_train[0]))

RESHAPED = 784

x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MNIST_V2')]

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, callbacks=callbacks, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)

print("\nTest score:", score[0])
print("Test accuracy:", score[1])