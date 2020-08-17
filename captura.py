import cv2

class Captura(object):
    def __init__(self, cameraIndex):
        self.camera = cv2.VideoCapture(cameraIndex)
