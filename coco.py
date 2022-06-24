"""
pull coco dataset from the internet with coco.sh
and study it to find relevant pixels

legend:
    gt = ground truth
    dt = detection


"""
import json
import os
from pprint import pprint
import random
from argparse import ArgumentParser as AP
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as io
import torch
from torchvision import transforms
from torchvision import utils as U
from torchvision.transforms import functional as F
from tqdm import tqdm
from transformers import YolosFeatureExtractor, YolosForObjectDetection, pipeline

import uutils as UU


def experiment1():
    """what percent of images are covered by ground truth boxes?"""

    dir, dset, annFile, coco = init_coco()

    # Load the categories in a variable
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    imgIds = coco.getImgIds()

    outliers = 0
    coverage_pcts = []
    for id in tqdm(imgIds):

        img = coco.loadImgs(id)[0]

        height, width = img["height"], img["width"]
        area = height * width

        I = io.imread(f'{coco_path}/images/{dset}/{img["file_name"]}') / 255.0

        annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        total = 0
        for ann in anns:
            total += round(ann["area"], 2)

        pct = round(total * 100 / area, 2)

        if pct > 100:
            outliers += 1
            # print(pct)
            # print([get_cat(ann['category_id'])['name'] for ann in anns])
            # plt.axis('off')
            # plt.imshow(I)
            # coco.showAnns(anns)
            # plt.show()

        coverage_pcts.append(pct)

    print(outliers)

    print(f"avg: {sum(coverage_pcts)/len(coverage_pcts)}")
    print(f"median: ")

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
    """
    if non ground truth areas are replaced with black pixels
    does the top 1 / top 5 accuracy change?
    for classification?

    in other words: are non ground truth pixels needed?
    """

    ap = AP()
    ap.add_argument('-r','--root',type=str)
    args = ap.parse_args()

    dset = 'val'
    coco_path = f"{args.root}/COCOdataset2017"
    coco = COCO( f"{coco_path}/annotations/instances_val2017.json")

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    imgIds = coco.getImgIds() # [:64]
    img_infos = coco.loadImgs(imgIds)

    # img2path = lambda img: f'{coco_path}/images/{dset}/{img["file_name"]}'
    # read_img = lambda img: plt.imread(img)
    # dataset = [img2path(img) for img in img_infos]

    # annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
    # anns = coco.loadAnns(annIds)

    # shadows = [f'{coco_path}/images/val_shadow/{img["file_name"]}' for img in img_infos]


    import matplotlib as mpl
    mpl.rcParams['figure.dpi']= 600

    for id, img in tqdm(zip(imgIds, img_infos), total=len(img_infos)):

        I = io.imread(f'{coco_path}/images/{dset}/{img["file_name"]}') / 255.0

        annIds = coco.getAnnIds(imgIds=id, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # anns = [(i["category_id"], i["bbox"]) for i in anns]

        # print(anns)
        # quit()

        if not anns:
            path = f'{coco_path}/images/val_shadow/{img["file_name"]}'
            plt.imsave(path, I, dpi=600)
            continue

        # print(anns)
        # quit()
        masks = [coco.annToMask(ann) for ann in anns]
        mask = np.sum(masks, axis=0)
        # fixes grayscale cuz it needs 3 channels
        m = ( I.shape[-1] if len(I.shape) == 3 else 1)  
        mask = np.array( [[[0, 0, 0] if col == 0 else [1, 1, 1] for col in row] for row in mask])
        I = np.array([[[col, col, col] for col in row] for row in I]) if m == 1 else I
        I *= mask

        path = f'{coco_path}/images/val_shadow/{img["file_name"]}'
        print(path)
        plt.imsave(path, I, dpi=600)
        quit()


def experiment2():
    """
    just do it for object detection instead
    cuz yolo is trained on coco not any resnet
    or mrcnn
    """

    ap = AP()
    ap.add_argument('-m','--models',type=str,nargs='+')
    ap.add_argument('-e','--eval',action='store_true')
    ap.add_argument('-i','--inference',action='store_true')
    ap.add_argument('-f','--file',type=str)
    ap.add_argument('-r','--root',type=str)
    args = ap.parse_args()

    dset = 'val'
    coco_path = f"{args.root}/COCOdataset2017"
    coco = COCO( f"{coco_path}/annotations/instances_val2017.json")

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)

    imgIds = coco.getImgIds() # [:64]
    img_infos = coco.loadImgs(imgIds)

    img2path = lambda img: f'{coco_path}/images/{dset}/{img["file_name"]}'
    read_img = lambda img: plt.imread(img)
    dataset = [img2path(img) for img in img_infos]

    annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    shadows = [f'{coco_path}/images/val_shadow/{img["file_name"]}' for img in img_infos]

    if args.inference: 

        model = "hustvl/yolos-base"
        pipe = pipeline("object-detection", model=model, device=-1)
        
        legend = UU.coco.id2l()
        l2id = lambda l: legend['l2id'][l]
        xyxy2xywh = lambda b: [b['xmin'],b['ymin'],b['xmax']-b['xmin'],b['ymax']-b['ymin']]

        for dset, filename in zip([dataset,shadows],['val_dt.json','val_shadows.json']):

            dt = []
            for i, item in tqdm(enumerate(dset), total=len(dset)):
                d = pipe(item, threshold=0.5)
                d = [
                    {
                        "score": x["score"],
                        "image_id": imgIds[i],
                        "category_id": l2id(x["label"]),
                        "bbox": xyxy2xywh(x['box']),
                    }
                    for x in d
                ]
                dt += d


            with open(filename,'w') as file:
                json.dump(dt,file)












        '''
        dt = []
        feature_extractor = YolosFeatureExtractor.from_pretrained("hustvl/yolos-tiny")
        model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")


        for id, p in tqdm(zip(imgIds, dataset), total=len(imgIds)):

            img = torch.Tensor(read_img(p))
            img = torch.stack([img,img,img],dim=-1) if len(img.shape) == 2 else img

            inputs = feature_extractor(img, return_tensors="pt")
            outputs = model(**inputs)
            bboxes = outputs.pred_boxes
            logits = outputs.logits

            outputs =  [UU.val.nms(bboxes=b, logits=l, size=img.shape) for b, l in zip(bboxes, logits)]
            for d in outputs:

                preds = [ {
                        "bbox": UU.bbox.xyxy2xywh(i[..., :4]).tolist(),
                        "score": float(i[..., 4].item()),
                        "category_id": int(i[..., -1].item()),
                        "image_id": id,
                    } for i in d ]
                dt += preds
            # pprint(len(preds))
            # quit()

        with open('val_dt.json', 'w') as file:
            json.dump(dt,file)
        '''

    if args.eval:

        for filename in ['val_dt.json','val_shadows.json']:

            print(filename)

            dt = coco.loadRes(filename)
            eval = COCOeval(coco, dt, "bbox")
            eval.params.imgIds  = imgIds

            eval.evaluate()
            eval.accumulate()
            eval.summarize()


def main():
    # experiment2()
    experiment2_mk_dataset()


if __name__ == "__main__":
    main()
