import cv2
import tensorflow as ft
from tensorflow import keras as k
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Input, Convolution2D, MaxPooling2D, Cropping2D, Conv2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random

params=[]
with open(r'mapa7/commands.txt') as realin:
    for line in realin:
        imgR = None
        par = line.split(';')
        timestamp = par[0]
        steering = par[2]
        imgRight = "mapa7/right/right-"+ str(timestamp) +".jpg"
        imgDepth = "mapa7/depth/depth-" + str(timestamp) +".jpg"
        params.append((imgRight,imgDepth, timestamp, steering ))

with open(r'dataset realinhamento/commands.txt') as realin:
    for line in realin:
        imgR = None
        par = line.split(';')
        timestamp = par[0]
        steering = par[2]
        imgRight = "dataset realinhamento/right/right-"+ str(timestamp) +".jpg"
        imgDepth = "dataset realinhamento/depth/depth-" + str(timestamp) +".jpg"
        params.append((imgRight,imgDepth, timestamp, steering ))

print('Loading dataset...')

def linhas(param):
    #img = cv2.imread(img, cv2.IMREAD_ANYCOLOR)
    img = cv2.imread(param, cv2.IMREAD_ANYCOLOR)
    df = np.diff(img,n=2, axis=-1)>10
    img[df[:,:,0]]=[0,0,0]
    img[~df[:,:,0]]=[255,255,255]
    img = img[200:,:,0]
    
    #img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img,(9,19),4)
    img = cv2.dilate(img,np.ones(3),iterations=8)
    img = cv2.erode(img,np.ones(3),iterations=8)
    
    return img


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def leiaImagens(lista):
    inputArray = np.zeros((len(lista), 280, 640, 2))
    outputArray = np.zeros((len(lista), 1))
    i=0
    for strImgR, strImgDep, timestamp, steering in lista:
        try:
            imgR = linhas(strImgR)
            imgDepth = cv2.imread(strImgDep, cv2.IMREAD_GRAYSCALE)[200:,:]

            imgArray = np.array([imgR, imgDepth])
            imgArray = np.moveaxis(imgArray, 0, 2)

            inputArray[i] = imgArray
            outputArray[i] = float(steering)
            i+=1
        except Exception as e:
            print('exception', e)

    inputArray = inputArray[:i]
    outputArray = outputArray[:i]
    return (inputArray, outputArray)


#model
model = Sequential()

#normalizing and cropping 
model.add(Input((280,640,2)))
model.add(Lambda(lambda x:x/255.0))
#model.add(Cropping2D(cropping=((206,72), (0,0)))) # 43%, 15%

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

model.summary()
model.compile(loss='mse', optimizer=k.optimizers.Adam(learning_rate=0.00005), metrics=ft.keras.metrics.RootMeanSquaredError())


model = k.models.load_model('seguidor2_gw3_com_mapa7_e_realinhamento.h5')
model.get_layer('conv2d').kernel_regularizer = ft.keras.regularizers.l2(0.001)
model.get_layer('conv2d_1').kernel_regularizer = ft.keras.regularizers.l2(0.001)
model.get_layer('conv2d_2').kernel_regularizer = ft.keras.regularizers.l2(0.001)

for x in range(200):
    print("x = ", x)
    for y in range(5):
        random.shuffle(params)
        #particoes = list(chunks(params, 500))
        #for e, p in enumerate(particoes):
        #print("part = ", e)
        #inp,out = leiaImagens(p)
        inp,out = leiaImagens(params[:700])
        model.fit(inp, out, epochs=40) # a,s,b
    model.save('seguidor2_gw3_com_mapa7_e_realinhamento.h5')

