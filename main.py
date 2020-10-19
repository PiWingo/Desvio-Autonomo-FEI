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

IM_WIDTH = 640
IM_HEIGHT = 480

leftData = None
vehicle = None
depthmap = None

script_dir = os.getcwd()

labels = open(script_dir + '/coco.names').read().strip().split("\n")

commands = open("./images/commands.txt", "a")

model = k.models.load_model(r'C:\Users\pietr\Desktop\carla9.10\PythonAPI\examples\seguidor2.h5')

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

def process_right(image):
	# timestamp = datetime.now().timestamp()
	# image.save_to_disk('./images/right/right-' + str(timestamp) + '.jpg')
	# global leftData, vehicle, depthmap
	# leftData.save_to_disk('./images/left/left-' + str(timestamp) + '.jpg')
	# cc = carla.ColorConverter.LogarithmicDepth
	# depthmap.save_to_disk('./images/depth/depth-' + str(timestamp) + '.jpg', cc)
	# controls = vehicle.get_control()
	# commands.write(str(timestamp) + ';' + str(controls.throttle) + ';' + str(controls.steer) + ';' + str(controls.brake) + '\n')

	
	if leftData != None and depthmap != None:
		global script_dir, labels, model
		
		rightImage = np.array(image.raw_data).reshape((IM_HEIGHT, IM_WIDTH, 4))
		rgbRightImage = rightImage[:, :, :3]
		grayRightImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)
		
		leftImage = np.array(leftData.raw_data).reshape((IM_HEIGHT, IM_WIDTH, 4))
		grayLeftImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
		
		# dephtMapImg = np.array(depthmap.raw_data).reshape((IM_HEIGHT, IM_WIDTH, 4))
		dephtMapImg = depth_to_logarithmic_grayscale(depthmap)

		imgProcess = ProcessamentoImagem(grayLeftImage, rgbRightImage, grayRightImage)

		# dephtMapImg = imgProcess.generate_depthmap()

		mockDephtMapImg = dephtMapImg[:, :, 0]
		# dephtMapImg = cv2.normalize(dephtMapImg, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		mockDephtMapImg = cv2.normalize(mockDephtMapImg, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		imgArray = np.array([grayRightImage, mockDephtMapImg])
		imgArray = np.expand_dims(imgArray, axis=0)
		imgArray = np.moveaxis(imgArray, 1, 3)
		if imgArray.shape == (1, 480, 640, 2):
			prediction = model.predict(imgArray)
			print(prediction)


		# classifiedImg = imgProcess.generate_bounding_boxes(script_dir+'/yolov3-tiny.cfg', script_dir+'/yolov3-tiny.weights', labels)
		cv2.imshow('dm', mockDephtMapImg)
		# cv2.imshow('left', leftImage[:, :, :3])
		cv2.imshow('right', grayRightImage)
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

	# f = open(r"C:\Users\pietr\Desktop\WindowsNoEditor\PythonAPI\examples\map.xodr", 'r')
	# xodr_xml = f.read()
	# f.close()

	# vertex_distance = 2.0   # in meters
	# max_road_length = 1000.0 # in meters
	# wall_height = 1.0       # in meters
	# extra_width = 0.6       # in meters
	# world = client.generate_opendrive_world(
    # 	xodr_xml, carla.OpendriveGenerationParameters(
    #     vertex_distance=vertex_distance,
    #     max_road_length=max_road_length,
    #     wall_height=wall_height,
    #     additional_width=extra_width,
    #     smooth_junctions=True,
    #     enable_mesh_visibility=True))

	world = client.get_world()

	world = client.load_world('Town07')

	blueprint_library = world.get_blueprint_library()
	for actor in world.get_actors():
		if actor.type_id == 'traffic.traffic_light':
			actor.destroy()
	# print(blueprint_library)
	bp = blueprint_library.filter('model3')[0]
	# kw = blueprint_library.filter('0010')[0]
	spawn_point1 = random.choice(world.get_map().get_spawn_points())
	spawn_point2 = carla.Transform(carla.Location(x=71.5, y=-5, z=.3), carla.Rotation(pitch=0.000000, yaw=-70, roll=0.000000))
	vehicle = world.spawn_actor(bp, spawn_point2)
	# motorcicle = world.spawn_actor(kw, spawn_point3)
	vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
	vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

	actor_list.append(vehicle)

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
