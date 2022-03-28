#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.preproc = ValTransform(legacy=False)


    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        
        img = img.cuda()
        img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            print("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def detect(img):
    
    exp = get_exp("./yolox/yolox_voc_l.py", "smoke_yolox_l")

    exp.test_conf = 0.5
    exp.nmsthre = 0.5
    exp.test_size = (640, 640)

    model = exp.get_model()

    model.cuda()
    model.half()  # to FP16
    model.eval()

    ckpt_file = "../weights/yolox_l_smoker_best.pth"
    
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    
    predictor = Predictor(
        model, exp, COCO_CLASSES
    )

    outputs, img_info = predictor.inference(img)

    if outputs[0] is None:
        print("YOLOX cannot find the cigarette.")
        return []

    bboxes = outputs[0][:, 0:4]
    ratio = img_info["ratio"]
    # preprocessing: resize
    bboxes /= ratio
    
    bboxes = bboxes.tolist()
    if len(bboxes) == 0:
        print("YOLOX cannot find the cigarette.")
        return bboxes
    else:
        boxes_final = [round(x) for x in bboxes[0]]
        # print(boxes_final)
        return boxes_final


if __name__ == "__main__":
    img = '../datasets/smoker_det/images/test/test_00110.jpg'
    box = detect(img)
    print(box)