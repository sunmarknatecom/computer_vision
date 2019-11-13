import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
import time
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import keras
from keras.layers import concatenate

#Time
start_time = time.time()

#data load
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
 
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255
 
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

 
 
# create model
input_img = Input(shape=(28, 28, 1), name='main_input')
 
 
 
#VGG Net
x1 = Conv2D(64, (3, 3))(input_img)
x1 = Activation('relu')(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64, (3, 3))(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D()(x1)
x1 = Flatten()(x1)
x1 = Dense(256)(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Dense(256)(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
 
 
#Res Net
x = Conv2D(64, (3, 3))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = (ZeroPadding2D((1,1)))(x)
x = Conv2D(64, (3, 3))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(1, (3, 3))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = (ZeroPadding2D((1,1)))(x)
x = concatenate([x, input_img], axis=2)
x = Flatten()(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
 
x = keras.layers.concatenate([x1, x])
out = Dense(num_classes, activation='softmax')(x)
 
# Compile model
model = Model(inputs=input_img, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
#model 가시화 만들기
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
 
 
 
# Fit the model
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=50, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
 
 
#모델 시각
fig, loss_ax = plt.subplots()
 
acc_ax = loss_ax.twinx()
 
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
 
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
 
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('accuracy')
 
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
 
plt.show()
 
#End Time
print("--- %s seconds ---" %(time.time() - start_time))
