'''
uses haar cascade not cnn
https://www.youtube.com/watch?v=mPCZLOVTEc4&list=PLzMcBGfZo4-lUA8uGjeXhBUUzPYc6vZRn&index=10
'''


import numpy as np
import cv2

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import utils as U
from torchvision import transforms

import time
import tkinter

import matplotlib.pyplot as plt
import matplotlib


print('running...')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# mrcnn.eval()

fcn = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
fcn = fcn.eval()


while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images


    results = yolo([frame])
    results.print()

    try:
        # print(results.__dict__.keys())
        print(len(results))
        (x,y,w,h,*_) = [int(x[0]) for x in results.xywh[0].numpy().T]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (225,0,0), 5)
        cv2.rectangle(frame, (y,x), (x+w, y+h), (225,0,0), 5)
    except IndexError:
        pass

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
