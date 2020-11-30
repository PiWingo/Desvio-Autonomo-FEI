# pylint: disable-all
import cv2
import numpy as np


class ProcessamentoImagem(object):

	left_image_grayscaled = None
	right_image_colored = None
	right_image_grayscaled = None

	def __init__(self, left_image_grayscaled, right_image, right_image_grayscaled):
		self.left_image_grayscaled = left_image_grayscaled
		self.right_image_colored = right_image
		self.right_image_grayscaled = right_image_grayscaled

	def generate_depthmap(self):
		win_size = 5
		min_disp = 0
		max_disp = 2*16
		num_disp = max_disp - min_disp  
		stereo = cv2.StereoSGBM_create(
			minDisparity=min_disp,
			numDisparities=3*16,
			blockSize=7,
			uniquenessRatio=5,
			speckleWindowSize=15,
			speckleRange=5,
			disp12MaxDiff=2,
			P1=8 * 3 * win_size ** 2,
			P2=32 * 3 * win_size ** 2,
		)
		return stereo.compute(self.left_image_grayscaled, self.right_image_grayscaled)

	def generate_bounding_boxes(self, configPath, weightsPath, labels):

		classifiedImg = self.right_image_colored

		(H, W) = classifiedImg.shape[:2]
		np.random.seed(42)
		COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		blob = cv2.dnn.blobFromImage(classifiedImg, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)

		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > .5:
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
		
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				color = [int(c) for c in COLORS[classIDs[i]]]
				text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
		return boxes, confidences, classIDs, idxs
