#Import Library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np
#import tflearn

#Akses hasil preprocessing
X =  pickle.load(open("train_XFix.pickle","rb"))
y =  pickle.load(open("label_yFix.pickle","rb"))
y = np.array(y)

X = X/255.0

Name = "Modelku{}".format(int(time.time()))

#TB = TensorBoard(log_dir = 'logs/{}'.format(Name))

model = Sequential()
#Model Neural Network
#1 Conv Layer
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
#Max Pooling
model.add(MaxPooling2D(pool_size=(7,7)))

#2 Conv Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
#Max Pooling
model.add(MaxPooling2D(pool_size=(7,7)))

#Flatten
model.add(Flatten())

#1 Fully Connected
model.add(Dense(128, input_shape = X.shape[1:]))
model.add(Activation("relu"))
#Dropout
model.add(Dropout(0.5))

#2 Fully Connected
model.add(Dense(128))
model.add(Activation("relu"))
#Dropout
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

#Mengatur Jumlah Epoch
#model.fit(X,y, batch_size = 1, epochs = 10, validation_split = 0.3, callbacks = [TB])
model.fit(X,y, batch_size = 1, epochs = 10, validation_split = 0.3)

#Menyimpan Model Neural Network
model.save('protoFix.model')

print("Sukses")
