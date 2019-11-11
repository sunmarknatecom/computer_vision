### Reference from: 모두의 딥러닝(2017, 길벗)

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

ds = np.loadtxt(input("Type the file name:\nex) ThoracicSurgery.csv\n>>>"), delimiter=",")

length_labels = int(len(ds[0])-1)

X = ds[:,:length_labels]
Y = ds[:,length_labels]

model = Sequential()
model.add(Dense(30, input_dim=length_labels, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10, verbose=0)

print("\n Accuracy: %02.2f%%" % (model.evaluate(X,Y)[1]*100))

# If authors wanted to delete the above codes, do it.
