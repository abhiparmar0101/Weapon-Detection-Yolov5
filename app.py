#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, Response

import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np

import torch
from io import BytesIO

DEVELOPMENT_ENV  = True

app = Flask(__name__)

app_data = {
    "name":         "Weapon Detection Template for a Flask Web App",
    "description":  "A basic Flask app using bootstrap for layout",
    "author":       "Weapon Detection Project",
    "html_title":   "Weapon-Detection",
    "project_name": "Weapon Detection Web-App",
    "keywords":     "flask, webapp, template, basic"
}



@app.route('/')
def index():
    return render_template('index.html', app_data=app_data)


def gen():
    # Load Custom Model
    # model = torch.hub.load("ultralytics/yolov5", "custom", path = "Weaponmodel/weapon.pt", force_reload=True)

    
    # model.eval()
    model.conf = 0.6  # confidence threshold (0-1)
    model.iou = 0.45  # NMS IoU threshold (0-1) 
    cap=cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            #print(results)
            #print(results.pandas().xyxy[0])
            #results.render()  # updates results.imgs with boxes and labels
            results.print()  # print results to screen
            #results.show() 
            #print(results.imgs)
            #print(type(img))
            #print(results)
            #plt.imshow(np.squeeze(results.render()))
            #print(type(img))
            #print(img.mode)
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

            #print(type(img))
            #print(img.shape)
            #frame = img
            #ret,buffer=cv2.imencode('.jpg',img)
            #frame=buffer.tobytes()
            #print(type(frame))
            #for img in results.imgs:
                #img = Image.fromarray(img)
            #ret,img=cv2.imencode('.jpg',img)
            #img=img.tobytes()

            #encode output image to bytes
            #img = cv2.imencode('.jpg', img)[1].tobytes()
            #print(type(img))
        else:
            break
        #print(cv2.imencode('.jpg', img)[1])

        #print(b)
        #frame = img_byte_arr

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_video_feed')
def webcam_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html', app_data=app_data)


def genvideo():
    modelvideo = torch.hub.load("ultralytics/yolov5", "custom", path = "Weaponmodel/weapon.pt", force_reload=False)

    modelvideo.eval()
    modelvideo.conf = 0.5  
    modelvideo.iou = 0.45 
    cap=cv2.VideoCapture('wd1.mp4')
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = modelvideo(img, size=640)
            results.print()  
            img = np.squeeze(results.render()) 
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        result1=b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        # print(result1)
        yield(result1)

@app.route('/video_feed')
def video_feed():
    return Response(genvideo(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/video')
def video():
    return render_template('video.html', app_data=app_data)






from PIL import Image
import datetime
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
@app.route('/images', methods=["GET", "POST"])
def predict():
    # model = torch.hub.load("ultralytics/yolov5", "custom", path = "Weaponmodel/weapon.pt", force_reload=True)
    # model.eval()
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        return redirect(img_savename)

    return render_template("images.html", app_data=app_data)
import subprocess
@app.route('/youtube')
def youtube():
    subprocess.run(['python3', 'yolov5/detect.py','--weights', 'yolov5/weapon.pt', '--source','https://youtu.be/3OSnluj-eDw'])
    return render_template('youtube.html', app_data=app_data)

if __name__ == '__main__':
    model = torch.hub.load("ultralytics/yolov5", "custom", path = "Weaponmodel/weapon.pt", force_reload=True)
    model.eval()
    app.run(debug=DEVELOPMENT_ENV)
