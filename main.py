# pylint: disable=line-too-long, mixed-indentation, missing-module-docstring, 
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
from processamento_imagem import ProcessamentoImagem
from matplotlib import pyplot as plt
from datetime import datetime
import tensorflow as ft
from tensorflow import keras as k
from keras.models import Sequential,Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Cropping2D, Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda

IM_WIDTH = 640
IM_HEIGHT = 480

leftData = None
middleData = None
vehicle = None
depthmap = None
markovMesh = {}

for x in range(0, 8):
	for y in range(0, 10):
		if y not in range(0, 3):
			markovMesh[(x, y)] = -0.7
		else:
			markovMesh[(x, y)] = -0.9

script_dir = os.getcwd()

labels = open(script_dir + '/coco.names').read().strip().split("\n")

#model
model = Sequential()

#normalizing and cropping 
model.add(Input((280,640,2)))
model.add(Lambda(lambda x:x/255.0))

#layer 1
model.add(Conv2D(32, (19,19), activation='relu', kernel_regularizer=ft.keras.regularizers.l2()))
model.add(MaxPooling2D(3,3))
model.add(Dropout(rate=0.3))
#layer 1
model.add(Conv2D(32, (15,15), activation='relu', kernel_regularizer=ft.keras.regularizers.l2()))
model.add(MaxPooling2D(3,3))
model.add(Dropout(rate=0.3))

#layer 2
model.add(Conv2D(32, (5,5), activation='relu', kernel_regularizer=ft.keras.regularizers.l2()))
model.add(MaxPooling2D(3,3))
model.add(Dropout(rate=0.3))

model.add(Flatten())

#fc 1
model.add(Dense(32, activation=ft.nn.relu))
model.add(Dropout(rate=0.3))
#fc 2
model.add(Dense(1,activation=ft.nn.tanh))

model.compile(loss='mse', optimizer=k.optimizers.Adam(learning_rate=0.00005), metrics=ft.keras.metrics.RootMeanSquaredError())

model.get_layer('conv2d').kernel_regularizer = ft.keras.regularizers.l2(0.001)
model.get_layer('conv2d_1').kernel_regularizer = ft.keras.regularizers.l2(0.001)
model.get_layer('conv2d_2').kernel_regularizer = ft.keras.regularizers.l2(0.001)

model.load_weights(r'C:\Users\pietr\Desktop\seguidor2_gw3_com_mapa7_e_realinhamento.h5') # use your own path to the keras model

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA np array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.int32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    grayscale = np.dot(array[:, :, :3], [256.0 * 256.0, 256.0, 1.0])
    grayscale /= (256.0 * 256.0 * 256.0 - 1.0)
    return grayscale


def depth_to_logarithmic_grayscale(image):
    """
    Convert an image containing CARLA encoded depth-map to a logarithmic
    grayscale image array.
    """
    grayscale = depth_to_array(image)
    # Convert to logarithmic depth.
    logdepth = np.ones(grayscale.shape) + (np.log(grayscale) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0)
    logdepth *= 255.0
    # Expand to three colors.
    return np.repeat(logdepth[:, :, np.newaxis], 3, axis=2)

def round_down(num, divisor):
    return num - (num%divisor)

def process_right(image):
	global leftData, vehicle, depthmap, middleData
	controls = vehicle.get_control()
	
	if leftData != None and depthmap != None and middleData != None:
		global script_dir, labels, model, markovMesh
		
		rightImage = np.array(image.raw_data).reshape((IM_HEIGHT, IM_WIDTH, 4))
		rgbRightImage = rightImage[:, :, :3]
		grayRightImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

		leftImage = np.array(leftData.raw_data).reshape((IM_HEIGHT, IM_WIDTH, 4))
		grayLeftImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
		
		dephtMapImg = np.array(depthmap.raw_data).reshape((IM_HEIGHT, IM_WIDTH, 4))
		dephtMapImg = depth_to_logarithmic_grayscale(depthmap)

		imgProcess = ProcessamentoImagem(grayLeftImage, rgbRightImage, grayRightImage)

		mockDephtMapImg = dephtMapImg[200:, :, 0]
		mockDephtMapImg = cv2.normalize(mockDephtMapImg, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		img = cv2.cvtColor(rightImage, cv2.IMREAD_ANYCOLOR)
		df = np.diff(rightImage,n=2, axis=-1)>10
		img[df[:,:,0]]=[0,0,0]
		img[~df[:,:,0]]=[255,255,255]
		img = img[200:,:,0]
		
		#img = cv2.equalizeHist(img)
		img = cv2.GaussianBlur(img,(9,19),4)
		img = cv2.dilate(img,np.ones(3),iterations=8)
		img = cv2.erode(img,np.ones(3),iterations=8)

		imgArray = np.array([img, mockDephtMapImg])
		imgArray = np.expand_dims(imgArray, axis=0)
		imgArray = np.moveaxis(imgArray, 1, 3)
		
		# if imgArray.shape == (1, 280, 640, 2):
		# 	prediction = model.predict(imgArray)
		# 	print('current => ' + str(controls.steer))
		# 	# print('current throttle => ' + str(controls.throttle))
		# 	print('prediction => ' + str(prediction[0][0]*-1.6))
		# 	if len(prediction) >= 0:
		# 		vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=float(prediction[0][0])*-1.6))

		boxes, confidences, classIDs, idxs = imgProcess.generate_bounding_boxes(script_dir+'/yolov3-tiny.cfg', script_dir+'/yolov3-tiny.weights', labels)
		
		if len(idxs) >= 1:
			vehicle.set_autopilot(False)
			for i in idxs.flatten():
				if labels[classIDs[i]] == 'person':
					(x, y) = (boxes[i][0], boxes[i][1])
					if y <= 320:
						(markovX, markovY) = (int(round_down(x, 64)/64), int(round_down(y, 40)/40))
						if y not in range(0, 3) and (markovX, markovY) in markovMesh:
							vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=markovMesh[(markovX, markovY)]))
		else:
			vehicle.set_autopilot(True)

		# print(boxes)
		# rgbRightImage[boxes[0][0], boxes[0][1]] = [0,0,255] 
		# cv2.imshow('dm', mockDephtMapImg)
		# cv2.imshow('left', leftImage[160:, :, :3])
		cv2.imshow('rightcrop', rightImage[160:, :, :3])
		cv2.imshow('right', rgbRightImage)
		cv2.waitKey(1)

def process_left(image):
	global leftData
	leftData = image

def depht_map(image):
	global depthmap
	depthmap = image

actor_list = []
try:
	client = carla.Client('127.0.0.1', 2000)
	client.set_timeout(2.0)

	world = client.get_world()

	world = client.load_world('Town04')

	blueprint_library = world.get_blueprint_library()
	bp = blueprint_library.filter('model3')[0]
	personBP = blueprint_library.filter('0010')[0]

	spawn_point2 = carla.Transform(carla.Location(x=-510.729706, y=175, z=0.281942), carla.Rotation(pitch=0.000000, yaw=90.357506, roll=0.000000))
	spawn_point3 = carla.Transform(carla.Location(x=-510.729706, y=190, z=0.281942), carla.Rotation(pitch=0.000000, yaw=90.357506, roll=0.000000))
	vehicle = world.spawn_actor(bp, spawn_point2)
	person = world.spawn_actor(personBP, spawn_point3)

	vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))
	tm = client.get_trafficmanager()
	tm.global_percentage_speed_difference(95)
	tm.ignore_lights_percentage(vehicle, 100)
	tm.collision_detection(vehicle, person, False)
	tm.auto_lane_change(vehicle, False)
	# vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

	actor_list.append(vehicle)
	actor_list.append(person)

	# https://carla.readthedocs.io/en/latest/cameras_and_sensors
	# get the blueprint for this sensor
	blueprint = blueprint_library.find('sensor.camera.rgb')
	blueprint_depthmap = blueprint_library.find('sensor.camera.depth')
	# change the dimensions of the image
	blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
	blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
	blueprint.set_attribute('fov', '110')

	blueprint_depthmap.set_attribute('image_size_x', f'{IM_WIDTH}')
	blueprint_depthmap.set_attribute('image_size_y', f'{IM_HEIGHT}')
	blueprint_depthmap.set_attribute('fov', '110')

	# Adjust sensor relative to vehicle
	spawn_point_left = carla.Transform(carla.Location(x=2.5, y=-0.5, z=1.2))
	spawn_point_right = carla.Transform(carla.Location(x=2.5, y=0.5, z=1.2))
	spawn_point_middle = carla.Transform(carla.Location(x=2.5, y=0, z=1.2))

	# spawn the sensor and attach to vehicle.
	right_camera = world.spawn_actor(blueprint, spawn_point_right, attach_to=vehicle)
	left_camera = world.spawn_actor(blueprint, spawn_point_left, attach_to=vehicle)
	depth_camera = world.spawn_actor(blueprint_depthmap, spawn_point_middle, attach_to=vehicle)

	# add sensor to list of actors
	actor_list.append(right_camera)
	actor_list.append(left_camera)
	actor_list.append(depth_camera)

	# do something with this sensor
	right_camera.listen(lambda data: process_right(data))
	left_camera.listen(lambda data: process_left(data))
	depth_camera.listen(lambda data: depht_map(data))

	while True:
		world.tick()
		
finally:
	print('destroying actors')
	for actor in actor_list:
		actor.destroy()
	print('done.')
