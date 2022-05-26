# Download COCO 2017:
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

mkdir -pv COCOdataset2017/annotations  COCOdataset2017/images/train COCOdataset2017/images/val

# Project Folder
# └─── *(.py / .ipynb)
# │
# └───COCOdataset2017
#     └───images
#     │   └───train
#     │   │    │   000000000009.jpg
#     │   │    │   000000000025.jpg
#     │   │    │   ...
#     │   └───val
#     │        │   000000000139.jpg
#     │        │   000000000285.jpg
#     │        │   ...
#     └───annotations
#         │   instances_train.json
#         │   instances_val.json
