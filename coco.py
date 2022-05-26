"""
pull coco dataset from the internet with coco.sh
and study it to find relevant pixels
"""
import json
import os
from pprint import pprint
import random

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io


def explore():
    with open("annotations/instances_val2017.json", "r") as file:
        data = json.load(file)
        [k for k in data.keys()]


def main():

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
            if cat['id'] == id:
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
    print("Number of images containing all the  classes:", len(imgIds),'\n')

    for id in imgIds:

        # load and display a random image
        rand = np.random.randint(0,len(imgIds))
        img = coco.loadImgs(imgIds[rand])[0]

        height = img['height']
        width = img['width']
        area = height*width

        print('img')
        print(f'{height}px * {width}px = {area}px2')
        print()


        I = io.imread(f'{dir}/images/{dset}/{img["file_name"]}')/255.0

        # plt.axis('off')
        plt.imshow(I)

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pprint([k for k in anns[0].keys()])

        total = 0
        for ann in anns:
            print(find_cat(ann['category_id'])['name'],f'\t\t ~area : {int(ann["area"])}')
            total += round(ann["area"],2)

        print()
        print(int(total))
        print(f'{round(total*100/area,2)}%')

    # coco.showAnns(anns)
    # plt.show()

if __name__ == "__main__":
    main()
