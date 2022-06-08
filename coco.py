"""
pull coco dataset from the internet with coco.sh
and study it to find relevant pixels
"""
import json
import os
from pprint import pprint
import random
from tqdm import tqdm

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io


def experiment1():
    '''what percent of images are covered by ground truth boxes?'''

    dir = "./COCOdataset2017"
    dset = "val"
    annFile = f"{dir}/annotations/instances_{dset}2017.json"

    # Initialize the COCO api for instance annotations
    coco = COCO(annFile)

    # Load the categories in a variable
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    def find_cat(id):
        for cat in cats:
            if cat["id"] == id:
                return cat
        return

    # print(cats)

    # # Define the classes (out of the 81) which you want to see. Others will not be shown.
    # filterClasses = ['laptop', 'tv', 'cell phone']
    # # Fetch class IDs only corresponding to the filterClasses
    # catIds = coco.getCatIds(catNms=filterClasses)
    # # Get all images containing the above Category IDs
    # imgIds = coco.getImgIds(catIds=catIds)

    imgIds = coco.getImgIds()
    print("Number of images containing all the  classes:", len(imgIds), "\n")

    outliers = 0
    coverage_pcts = []
    for id in tqdm(imgIds):

        # load and display a random image
        # rand = np.random.randint(0,len(imgIds))
        # img = coco.loadImgs(imgIds[rand])[0]
        img = coco.loadImgs(id)[0]

        height = img["height"]
        width = img["width"]
        area = height * width

        # print('img')
        # print(f'{height}px * {width}px = {area}px2')
        # print()

        I = io.imread(f'{dir}/images/{dset}/{img["file_name"]}') / 255.0


        annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pprint([k for k in anns[0].keys()])

        total = 0
        for ann in anns:
            # print(find_cat(ann['category_id'])['name'],f'\t\t ~area : {int(ann["area"])}')
            total += round(ann["area"], 2)

        # print()
        # print(int(total))
        pct = round(total * 100 / area, 2)
        # print(f'{pct}%')

        if pct > 100:
            outliers += 1
            # print(pct)
            # print([find_cat(ann['category_id'])['name'] for ann in anns])
            # plt.axis('off')
            # plt.imshow(I)
            # coco.showAnns(anns)
            # plt.show()

        coverage_pcts.append(pct)

    print(outliers)
    # print(coverage_pcts)
    print(f"avg: {sum(coverage_pcts)/len(coverage_pcts)}")
    print(f'median: ')

    fig, ax = plt.subplots()
    ax.hist(coverage_pcts, bins=20)
    ax.set(
        title="Images in COCO2017 val dataset",
        xlabel="% of image covered by ground truth pixels",
        ylabel="frequency",
    )
    ax.set_xlim([0, max(coverage_pcts)])

    plt.show()

def experiment2_mk_dataset():
    '''
    if non ground truth areas are replaced with black pixels
    does the top 1 / top 5 accuracy change?
    for classification?

    in other words: are non ground truth pixels needed?
    '''

    dir = "./COCOdataset2017"
    dset = "val"
    annFile = f"{dir}/annotations/instances_{dset}2017.json"

    # Initialize the COCO api for instance annotations
    coco = COCO(annFile)

    # Load the categories in a variable
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    def find_cat(id):
        for cat in cats:
            if cat["id"] == id:
                return cat
        return

    imgIds = coco.getImgIds()
    imgs = coco.loadImgs(imgIds)

    # outliers = 0
    # coverage_pcts =

    for id,img in tqdm(zip(imgIds,imgs), total=len(imgs)):

        height = img["height"]
        width = img["width"]
        area = height * width

        I = io.imread(f'{dir}/images/{dset}/{img["file_name"]}') / 255.0

        annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # points = []
        # for ann in anns:
        #     shape = ann['segmentation'][0]
        #     points += [(xi,yi) for xi,yi in zip(shape[::2],shape[1::2])]
        # quit()

        # total = 0
        # for ann in anns:
        #     # print(find_cat(ann['category_id'])['name'],f'\t\t ~area : {int(ann["area"])}')
        #     total += round(ann["area"], 2)

        # print()
        # print(int(total))
        # pct = round(total * 100 / area, 2)
        # print(f'{pct}%')

        # print([find_cat(ann['category_id'])['name'] for ann in anns])

        # fig,axs = plt.subplots(1,2)
        # plt.axis('off')

        # axs[0].imshow(I)
        # plt.imshow(I), plt.show()
        # coco.showAnns(anns)
        if not anns:
            path = f'{dir}/images/val_ground_truth/{img["file_name"]}'
            plt.imsave(path,I)
            continue

        masks = [coco.annToMask(ann) for ann in anns]
        mask = np.sum(masks, axis=0)
        m = I.shape[-1] if len(I.shape) == 3 else 1 # fixes grayscale cuz it needs 3 channels
        mask = np.array([[ [0,0,0] if col == 0 else [1,1,1] for col in row] for row in mask])
        I = np.array([[[col,col,col]for col in row] for row in I]) if m == 1 else I
        I *= mask

        # axs[1].imshow(I)

        path = f'{dir}/images/val_ground_truth/{img["file_name"]}'
        plt.imsave(path,I)

        # if input('next? (y/n) ') == 'n':
        #     quit()

        # plt.scatter([i[0] for i in points],[i[1] for i in points])

        # plt.show()

def experiment2():
    '''
    just do it for object detection instead
    cuz yolo is trained on coco not any resnet
    or mrcnn
    '''

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # mrcnn.eval()

    for dset in ('val', 'val_ground_truth'):

        '''
        clean up repetitive code
        def init_dset()
        doesnt need to be 250 lines
        '''

        imgs = []
        for img in imgs:

            results = yolo([img])
            results.print()

            quit() # just do one inference to start
        quit() # just do one inference to start

    try:
        # print(results.__dict__.keys())
        print(len(results))
        (x,y,w,h,*_) = [int(x[0]) for x in results.xywh[0].numpy().T]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (225,0,0), 5)
        cv2.rectangle(frame, (y,x), (x+w, y+h), (225,0,0), 5)
    except IndexError:
        pass


def main():
    experiment2()

if __name__ == "__main__":
    main()
