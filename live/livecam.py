
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

def show(imgs, *, pic=None, classes=[]):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        try:
            img = img.detach()
        except:
            pass

        if type(pic) != type(None):
            mask = np.array([ [ x if x < 0.5 else np.nan for x in y] for y in img])
            axs[0, i].imshow(np.asarray(pic))
        # img = F.to_pil_image(img)
        # axs[0, i].imshow(np.asarray(img) *kwargs)
        if classes:
            axs[0, i].set(title=f'{classes[i]}')
        axs[0, i].imshow(mask, cmap='binary',alpha=0.8)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


    plt.show()
    quit()


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

    # # def segment():
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # transform = transforms.Compose([transforms.ToTensor()])
    # tensor = transform(image)
    #
    #
    # batch_int = torch.stack([tensor])
    # batch = F.convert_image_dtype(batch_int, dtype=torch.float)
    #
    # norm_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #
    # print(norm_batch.shape)
    #
    # output = fcn(norm_batch)['out']
    #
    # print(output.shape, output.min().item(), output.max().item())
    #
    # sem_classes = [
    #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    #
    # normalized_masks = torch.nn.functional.softmax(output, dim=1)
    #
    # classes = ['person']
    # masks = [
    #     normalized_masks[img_idx, sem_class_to_idx[cls]]
    #     for img_idx in range(batch.shape[0])
    #     for cls in classes
    # ]
    #
    # # print(len(masks))
    #
    # show(masks, pic=image, classes=classes)

    # segment()

    results = yolo([frame])
    results.print()


    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (225,0,0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+w, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
