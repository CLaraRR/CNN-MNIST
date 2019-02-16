import numpy as numpy
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

nb_class = 10
nb_epoch = 4
batchsize = 128

# prepare your data mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() # download default in .keras/datasets/mnist.npz
                                                         # mnist.load_data(.keras/datasets/mnist.npz)
# setup data shape
x_train = x_train.reshape(-1, 28, 28, 1)  # -1:sample number unknow, 28*28:sample size,1:single channel
x_test = x_test.reshape(-1, 28, 28, 1)  # if Theao (1,28,28)

# one-hot
y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)

# setup model
model = Sequential()

# 1st conv2d layer
model.add(Convolution2D(
        filters = 32, 
        kernel_size = [5, 5], 
        padding = 'same', 
        input_shape = (28, 28, 1)
))

model.add(Activation('relu')) # often relu

model.add(MaxPool2D(
    pool_size=(2, 2), 
    strides=(2, 2), 
    padding='same'
))

# 2nd conv2d layer
model.add(Convolution2D(
    filters=64, 
    kernel_size=(5, 5), 
    padding='same'
))

model.add(Activation('relu'))

model.add(MaxPool2D(
    pool_size=(2, 2), 
    strides=(2, 2), 
    padding='same'
))

# 1st fully connected dense
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# 2nd fully connected dense
model.add(Dense(10))
model.add(Activation('softmax')) # the last activation function definitely is softmax

# define optimizer and setup param
adam = Adam(lr=0.0001)

# compile model
model.compile(
    optimizer = adam,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# run network
model.fit(
    x = x_train,
    y = y_train,
    epochs = nb_epoch,
    batch_size = batchsize,
    validation_data = (x_test,y_test)
)
model.save('model')

new_model = load_model('model')
(loss, accuracy) = new_model.evaluate(x_test, y_test)
print('loss is:=>>' , loss)
print('accuracy is:=>>' , accuracy)


