!pip install tensorflow keras mnist matplotlib numpy

#importing packages
import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential #ANN architecture
from keras.layers import Dense # thelayers of the ANN
from keras.utils import to_categorical

#loading data
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#Normalising the data- the pixel values from [0,255] to [-0.5,0.5] to make our network easier to train
train_images = (train_images/255)-0.5
test_images = (test_images/255)-0.5

#Flatten the images from 28*28 to 784 dimensional vector to pass into the neural network
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))

#printing shape of the images
print(train_images.shape) # 60000 rows and 784 columns
print(test_images.shape) # 10000 rows and 784 columns

#building the model of total 3 layers, 2 layers with 64 neurons and relu func and last layer with 10 neuron with softmax func
model = Sequential()
model.add( Dense(64, activation='relu', input_dim=784))
model.add( Dense(64, activation='relu'))
model.add( Dense(10, activation='softmax'))

#compile the model
#the loss function shows how well the model did on training and then tries to improve on it using the optimiser
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # allows more than two classes
    metrics=['accuracy']
)

#train the model
model.fit(
    train_images,
    to_categorical(train_labels), #2 then it shows [0,0,1,0,0,0,0,0,0,0]
    epochs = 5,
    batch_size=32  #no of samples per gradient update for training 
)

#Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)
