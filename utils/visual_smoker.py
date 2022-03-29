from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import matplotlib.pyplot as plt

import pickle
import json
import numpy as np
import cv2
import os
import sys
import argparse

import matplotlib as mpl
mpl.use('Agg')

obj_list = ['person', 'cigarette']

# 根据类别id获取对应的hoi类别
with open("./datasets/smoker_det/annotations/hoi_list.json", "r") as file:
    hois = json.load(file)
num_hois = len(hois)
union_action_list = {}
for i, item in enumerate(hois):
    union_action_list[i] = item["verb"] + "_" + item["object"]


def visual_smoker_demo(preds_inst, detection, im_path, save_path):  
    dpi = 80
    im_data = plt.imread(im_path)
    im_name = im_path.split('/')[-1]
    im_id = im_name.split('.')[0]

    if len(im_data.shape) == 2:
        print("the test image is 1 channnel.")
        return
    height, width, nbands = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, interpolation='nearest')

    # visual the instance detection
    num_inst = len(preds_inst["rois"])
    if num_inst == 1:
        box = preds_inst["rois"][0]
        ax.add_patch(
            plt.Rectangle((box[0], box[1]),
                          box[2] - box[0],
                          box[3] - box[1], fill=False,
                          edgecolor="red", linewidth=2)
        )
        text = obj_list[preds_inst["obj_class_ids"][0]] + " ," + "%.3f"%preds_inst["obj_scores"][0]
        ax.text(box[0] + 5, box[1] + 10,
                text, fontsize=10, color='blue')
    else:
        for inst_id in range(2):
            box = preds_inst["rois"][inst_id]
            print("box", box)
            ax.add_patch(
                plt.Rectangle((box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1], fill=False,
                            edgecolor="red", linewidth=2)
            )
            text = obj_list[preds_inst["obj_class_ids"][inst_id]] + " ," + "%.3f"%preds_inst["obj_scores"][inst_id]
            ax.text(box[0] + 5, box[1] + 10,
                    text, fontsize=10, color='blue')
    fig.savefig(os.path.join(save_path, "%s_instances.jpg"%im_id))
    plt.close()

    # 处理交互的标签
    for ele_id, ele in enumerate(detection[:1]):
        # only pick top-1 to visualize
        role_scores = ele[3] 
        role_scores_idx_sort = np.argsort(role_scores)[::-1]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(im_data, interpolation='nearest')

        H_box = ele[0]
        O_box = ele[1]
       
        ax.add_patch(
            plt.Rectangle((H_box[0], H_box[1]),
                          H_box[2] - H_box[0],
                          H_box[3] - H_box[1], fill=False,
                          edgecolor="red", linewidth=2)
        )

        ax.add_patch(
            plt.Rectangle((O_box[0], O_box[1]),
                          O_box[2] - O_box[0],
                          O_box[3] - O_box[1], fill=False,
                          edgecolor="green", linewidth=2)
        )

        # 绘制预测的top-3类别
        for action_count in range(3):
            text = union_action_list[role_scores_idx_sort[action_count]] + ", %.2f" % role_scores[role_scores_idx_sort[action_count]]
            print("pred res top", action_count+1, ":", text)
         
        text = union_action_list[role_scores_idx_sort[0]]
        if text == "in_hand_and_mouth_cigarette":
            text = "cigarette in hand and mouth"
        elif text == "in_hand_cigarette":
            text = "cigarette in hand"
        else:
            text = "cigarette in mouth"

        ax.text(H_box[0] + 5, H_box[1] + 10, text, fontsize=10, color='blue')
        fig.savefig(os.path.join(save_path, "%s_hoi.jpg"%im_id))

        plt.close()



if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--det_file', type=str, default=None)
    args = arg.parse_args()

    detection = pickle.load(open(args.det_file, "rb"))
    visual_smoker_demo(detection)
