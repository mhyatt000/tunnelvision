import time
import tkinter

from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision import utils as U
from torchvision.transforms import functional as F
from transformers import YolosFeatureExtractor, YolosForObjectDetection

import uutils as UU


def livecam(*, detector=None, backbone=None):
    print('running...')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # shape is 720x1280 hxw for your mac 

        resize = True
        if resize:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        if detector and backbone:

            inputs = backbone(images=frame, return_tensors="pt")
            outputs = detector(**inputs)

            print(outputs.loss)
            logits = outputs.logits #.detach().reshape((100,92))
            bboxes = outputs.pred_boxes #.detach().reshape((100,4))
            index = [i for i in range(100)]

            threshold = 0.5
            probas = outputs.logits.softmax(-1)[0, :, :-1].cpu()
            keep = probas.max(-1).values > threshold
            vis_indexs = torch.nonzero(keep).squeeze(1)
            
            UU.tools.show_pred_fig(frame, outputs, keep)

        else:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_bbox_in_img(fname, bbox_scaled, score, color=[0,255,0]):
    tl = 3
    tf = max(tl-1,1) # font thickness
    # color = [0,255,0]
    im = cv2.imread(fname)
    for p, (xmin, ymin, xmax, ymax) in zip(score, bbox_scaled.tolist()):
        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(im, c1, c2, color, tl, cv2.LINE_AA)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(fname, im)


def main():
    'main'

    backbone = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    yolos = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

    print('loaded yolos')

    livecam(detector=yolos, backbone=backbone)


if __name__ == '__main__':
    main()
