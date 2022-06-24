import json

from tqdm import tqdm
import uutils as UU

with open("val_dt.json") as file:
    data = json.load(file)

legend = UU.coco.id2l()
id2l = lambda id: legend["id2l"][id].replace(' ','_')

ids = set([d["image_id"] for d in data])
for id in tqdm(ids):
    dt = [d for d in data if d["image_id"] == id]

    "xywh"
    xywh2xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]

    boxes = [d["bbox"] for d in dt]
    boxes = [xywh2xyxy(b) for b in boxes]
    boxes = [[str(int(x)) for x in b] for b in boxes]


    dt = [ " ".join([id2l(d["category_id"]), str(d["score"]), *b]) for d, b in zip(dt, boxes) ]
    with open(f"{id}.txt", "w") as file:
        [file.write(l + "\n") for l in dt]
