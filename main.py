from flask import Flask, render_template, Response
# from flask_socketio import SocketIO, emit
from typing import Counter
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
clasificador = Classifier("Models/keras_model.h5","Models/labels.txt")
global pedido

offset=20
imgSize=300

folder = "Datos/A"
counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

app = Flask(__name__)
# app.config['SECRET_KEY'] = "secret:"
# socketio = SocketIO(app)
# print(socketio)

# @socketio.on('message')
# def mostrarMensaje(msg):
#     print("message: " + msg)
#     emit('message', msg)

def removeDuplicates(s):
    chars = []
    prev = None
 
    for c in s:
        if prev != c:
            chars.append(c)
            prev = c
 
    return ''.join(chars)

def generar_frame():
    global pedido
    pedido = "NO TIENE PEDIDO"
    palabras = ""
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        imagensalida = img.copy()
        hands, img = detector.findHands(img)
        cv2.putText(imagensalida,pedido,(10,30),cv2.FONT_HERSHEY_COMPLEX,0.6,(20, 56, 167),1)
        if hands:
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
                        palabras = palabras + str(labels[index])
                        print(prediccion[index])
                        pedido = removeDuplicates(palabras)
                        cv2.putText(imagensalida,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                        cv2.rectangle(imagensalida,(x - offset, y - offset),(x + w + offset,y + h + offset),(255,0,255),4)
                except:
                    print("Se salio fuera de los margenes de la camara")
            else:
                try:
                    k = imgSize/w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                    hGap = math.ceil((imgSize-hCal)/2)
                    imgWhite[hGap:hCal+hGap,:] = imgResize
                    prediccion, index = clasificador.getPrediction(imgWhite,draw=False)
                    if prediccion[index] > 0.90:
                        palabras = palabras + str(labels[index])
                        print(prediccion[index])
                        pedido = removeDuplicates(palabras)
                        cv2.putText(imagensalida,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                        cv2.rectangle(imagensalida,(x - offset, y - offset),(x + w + offset,y + h + offset),(255,0,255),4)
                except:
                    print("Se salio fuera de los margenes de la camara")
        if not success:
            break
        else:
            suc, encode = cv2.imencode('.jpg',imagensalida)
            frame = encode.tobytes()
            
        yield(b'--frame\r\n'
            b'content-Type: image/jpeg\r\n\r\n'+frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generar_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pagina')
def pagina():
    return "tengo el pedido de: {}".format(pedido)

if __name__ == "__main__":
    app.run(debug=True)
    # socketio.run(app)