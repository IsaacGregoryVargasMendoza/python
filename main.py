from flask import Flask, render_template, Response, request
# from flask_socketio import SocketIO, emit
from typing import Counter
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import cv2
import mediapipe as mp
import base64
import io, base64
# from PIL import Image, ImageOps
# from io import BytesIO
import os
# from keras.models import load_model

# detector = HandDetector(maxHands=1)
detector = HandDetector()
clasificador = Classifier(os.path.dirname(__file__) + "/Models/keras_model.h5",os.path.dirname(__file__) + "/Models/labels.txt")

# model = load_model("keras_Model.h5", compile=False)
# class_names = open("labels.txt", "r").readlines()
global pedido

offset=20
imgSize=300

# folder = "Datos/A"
# counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

app = Flask(__name__)

def removeDuplicates(s):
    chars = []
    prev = None
 
    for c in s:
        if prev != c:
            chars.append(c)
            prev = c
 
    return ''.join(chars)

# def generar_frame():
#     cap = cv2.VideoCapture(0)
#     global pedido
#     pedido = "NO TIENE PEDIDO"
#     palabras = ""
#     while True:
#         success, img = cap.read()
#         img = cv2.flip(img,1)
#         imagensalida = img.copy()
#         hands, img = detector.findHands(img)
#         cv2.putText(imagensalida,pedido,(10,30),cv2.FONT_HERSHEY_COMPLEX,0.6,(20, 56, 167),1)
#         if hands:
#             hand = hands[0]
#             x,y,w,h = hand['bbox']

#             imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
#             imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

#             aspectRatio = h/w

#             if aspectRatio>1:
#                 try:
#                     k = imgSize/h
#                     wCal = math.ceil(k * w)
#                     imgResize = cv2.resize(imgCrop,(wCal,imgSize))
#                     wGap = math.ceil((imgSize-wCal)/2)
#                     imgWhite[:,wGap:wCal+wGap] = imgResize
#                     prediccion, index = clasificador.getPrediction(imgWhite,draw=False)
#                     if prediccion[index] > 0.90:
#                         palabras = palabras + str(labels[index])
#                         print(prediccion[index])
#                         pedido = removeDuplicates(palabras)
#                         cv2.putText(imagensalida,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
#                         cv2.rectangle(imagensalida,(x - offset, y - offset),(x + w + offset,y + h + offset),(255,0,255),4)
#                 except:
#                     print("Se salio fuera de los margenes de la camara")
#             else:
#                 try:
#                     k = imgSize/w
#                     hCal = math.ceil(k * h)
#                     imgResize = cv2.resize(imgCrop,(imgSize,hCal))
#                     hGap = math.ceil((imgSize-hCal)/2)
#                     imgWhite[hGap:hCal+hGap,:] = imgResize
#                     prediccion, index = clasificador.getPrediction(imgWhite,draw=False)
#                     if prediccion[index] > 0.90:
#                         palabras = palabras + str(labels[index])
#                         print(prediccion[index])
#                         pedido = removeDuplicates(palabras)
#                         cv2.putText(imagensalida,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
#                         cv2.rectangle(imagensalida,(x - offset, y - offset),(x + w + offset,y + h + offset),(255,0,255),4)
#                 except:
#                     print("Se salio fuera de los margenes de la camara")
#         if not success:
#             break
#         else:
#             suc, encode = cv2.imencode('.jpg',imagensalida)
#             frame = encode.tobytes()
            
#         yield(b'--frame\r\n'
#             b'content-Type: image/jpeg\r\n\r\n'+frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/obtener-letra', methods=['POST'])
def obtenerLetra():
    print("recibi la imagen")
    # cap = base64.b64decode(request.json['imagen'].replace("data:image/png;base64,",""))
    decoded_bytes = base64.b64decode(request.json['imagen'].split(",")[1])

    # imgAux = Image.open(io.BytesIO(decoded_bytes))

    # np_array = np.asarray(imgAux, dtype=np.uint8)

    # print(np_array)

    filename = os.path.dirname(__file__) + '/some_image.jpg'

    with open(filename, 'wb') as f:
        f.write(decoded_bytes)

    img = cv2.imread(filename)
    img = cv2.flip(img, 1)

    # img = cv2.flip(np_array,1)

    # imagensalida = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        print("entre a Hands")
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = h/w

        if aspectRatio>1:
            try:
                k = imgSize/h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:,wGap:wCal+wGap] = imgResize
                prediccion, index = clasificador.getPrediction(imgWhite,draw=False)
                if prediccion[index] > 0.90:
                    # palabras = "palabras" + str(labels[index])
                    print(prediccion[index])
                    # pedido = removeDuplicates(palabras)
                    return labels[index]
                    # cv2.putText(imagensalida,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                    # cv2.rectangle(imagensalida,(x - offset, y - offset),(x + w + offset,y + h + offset),(255,0,255),4)
            except:
                print("Se salio fuera de los margenes de la camara")
                return "-1"
        else:
            try:
                k = imgSize/w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap,:] = imgResize
                prediccion, index = clasificador.getPrediction(imgWhite,draw=False)
                if prediccion[index] > 0.90:
                    # palabras = "palabras" + str(labels[index])
                    print(prediccion[index])
                    # pedido = removeDuplicates(palabras)
                    return labels[index]
                    # cv2.putText(imagensalida,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                    # cv2.rectangle(imagensalida,(x - offset, y - offset),(x + w + offset,y + h + offset),(255,0,255),4)
            except:
                print("Se salio fuera de los margenes de la camara")
                return "-1"
    return "-1"

@app.route('/pagina')
def pagina():
    # return "tengo el pedido de: {}".format(pedido)
    return "tengo el pedido de: "

if __name__ == "__main__":
    app.run(debug=False)
    # socketio.run(app)