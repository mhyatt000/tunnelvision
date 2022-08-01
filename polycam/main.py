"apply polycam on an inference"

from argparse import ArgumentParser as AP

from datasets import load_dataset
import torch
from transformers import AutoFeatureExtractor, ResNetForImageClassification


def get_args():
    """returns arguments"""

    ap = AP()
    args = ap.parse_args()
    return args


def main():

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    pipe = transformers.pipeline("image-classification", model="microsoft/resnet-50")
    model = pipe.model
    print(type(model))

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])


if __name__ == "__main__":
    main()
