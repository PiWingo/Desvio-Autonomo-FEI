import cv2
import tensorflow as ft
from tensorflow import keras as k
import matplotlib.pyplot as plt
import numpy as np

f = open(r'C:\Users\pietr\Desktop\DATASETS\DATASET2\commands.txt')
allLines = f.read().splitlines()
inputArray = []
outputArray = []

print('Loading dataset...')

for line in allLines:

	imgL = None
	imgR = None

	params = line.split(';')
	
	timestamp = params[0]
	acceleration = params[1]
	steering = params[2]
	breakValue = params[3]

	try:
		imgR = cv2.imread(r"C:\Users\pietr\Desktop\DATASETS\DATASET2\right\right-"+ timestamp +".jpg", cv2.IMREAD_GRAYSCALE)
		imgDepth = cv2.imread(r"C:\Users\pietr\Desktop\DATASETS\DATASET2\depth\depth-" + timestamp +".jpg", cv2.IMREAD_GRAYSCALE)

		imgArray = np.array([imgR, imgDepth])
		imgArray = np.moveaxis(imgArray, 0, 2)

		inputArray.append(imgArray)
		outputArray.append(np.array([float(acceleration), float(steering), float(breakValue)]))

	except:
		print('error loading: ' + timestamp)

print('Dataset loaded')

print('Validating dataset')

for i in range(0,len(inputArray)):
	print('index number ' + str(i))
	print('index shape ' + str(inputArray[i].shape))
	# print('index output value ' + str(outputArray[i]))


print('inputArray size: ' + str(len(inputArray)))
print('outputArray size: ' + str(len(outputArray)))
print(inputArray[0].shape)
print(outputArray[0])

print('Executing model instructions...')
inp = k.layers.Input([480,640,2])
l1 = k.layers.Convolution2D(4, 7, 7, data_format='channels_last')(inp)
l2 = k.layers.Flatten()(l1)
l3 = k.layers.Dense(3)(l2)
model = k.Model(inp, l3)
model.summary()
model.compile(loss=k.losses.MeanSquaredError())
print('Model instructions finished')
print('Starting training...')
print(np.array(inputArray))
model.fit(np.array(inputArray), np.array(outputArray)) # a,s,b
print('Finished training...')
print('Exporting model...')
model.save('seguidor.h5')
print('Script finished')