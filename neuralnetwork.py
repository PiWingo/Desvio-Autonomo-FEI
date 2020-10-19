import cv2
import tensorflow as ft
from tensorflow import keras as k
from keras.models import Sequential,Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Cropping2D, Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
import matplotlib.pyplot as plt
import numpy as np

mapa4 = open(r'C:\Users\pietr\Desktop\DATASETS\mapa4\commands.txt')
mapa7 = open(r'C:\Users\pietr\Desktop\DATASETS\mapa7\commands.txt')
allLines4 = mapa4.read().splitlines()
allLines7 = mapa7.read().splitlines()
inputArray = []
outputArray = []
mapa4.close()
mapa7.close()

print('Loading dataset...')

for line in allLines4:
	imgR = None
	params = line.split(';')
	timestamp = params[0]
	steering = params[2]
	try:
		imgR = cv2.imread(r"C:\Users\pietr\Desktop\DATASETS\mapa4\right\right-"+ timestamp +".jpg", cv2.IMREAD_GRAYSCALE)
		imgDepth = cv2.imread(r"C:\Users\pietr\Desktop\DATASETS\mapa4\depth\depth-" + timestamp +".jpg", cv2.IMREAD_GRAYSCALE)

		imgArray = np.array([imgR, imgDepth])
		imgArray = np.moveaxis(imgArray, 0, 2)

		inputArray.append(imgArray)
		outputArray.append(np.array([float(steering)]))

	except:
		print('mapa4 error loading: ' + timestamp)

print('map4 input size: ' + str(len(inputArray)))

for line in allLines7:
	imgR = None
	params = line.split(';')
	timestamp = params[0]
	steering = params[2]
	try:
		imgR = cv2.imread(r"C:\Users\pietr\Desktop\DATASETS\mapa7\right\right-"+ timestamp +".jpg", cv2.IMREAD_GRAYSCALE)
		imgDepth = cv2.imread(r"C:\Users\pietr\Desktop\DATASETS\mapa7\depth\depth-" + timestamp +".jpg", cv2.IMREAD_GRAYSCALE)

		imgArray = np.array([imgR, imgDepth])
		imgArray = np.moveaxis(imgArray, 0, 2)

		inputArray.append(imgArray)
		outputArray.append(np.array([float(steering)]))

	except:
		print('mapa7 error loading: ' + timestamp)

print('Dataset loaded')

print('Validating dataset')

print('inputArray size: ' + str(len(inputArray)))
print('outputArray size: ' + str(len(outputArray)))

print('Test' + str(np.array(inputArray[0]).shape))

print('Executing model instructions...')


#model
model = Sequential()

#normalizing and cropping 
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(480,640,2)))
# model.add(Cropping2D(cropping=((206,72), (0,0)))) # 43%, 15%

#layer 1
model.add(Conv2D(32, (7,7), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.3))

#layer 2
model.add(Conv2D(32, (7,7), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.3))

model.add(Flatten())

#fc 1
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.3))
#fc 2
model.add(Dense(32, activation='relu'))
#fc 3
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
print(np.array(inputArray).shape)
print(np.array(outputArray).shape)
model.fit(np.array(inputArray), np.array(outputArray)) # a,s,b
# model.save('seguidor2.h5')

# inp = k.layers.Input([480,640,2])
# l1 = k.layers.Convolution2D(4, 7, 7, data_format='channels_last')(inp)
# l2 = k.layers.Flatten()(l1)
# l3 = k.layers.Dense(3)(l2)
# model = k.Model(inp, l3)
# model.summary()
# model.compile(loss=k.losses.MeanSquaredError())
# print('Model instructions finished')

# print('Starting training...')
# # model.fit(np.array(inputArray), np.array(outputArray)) # a,s,b
# print('Finished training')

# print('Exporting model...')
# model.save('seguidor.h5')
# print('Script finished')