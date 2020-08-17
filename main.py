import Captura

if __name__ == "__main__" :
	cam1 = Captura(1)

	while True:

		ret0, frame0 = cam1.camera.read()
