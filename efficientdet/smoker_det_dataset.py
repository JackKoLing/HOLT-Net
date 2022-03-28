import os
import torch
import json
import numpy as np
import sys
import copy
sys.path.append("./")

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps, ImageFile

import cv2
import random
from tqdm.autonotebook import tqdm

from efficientdet.vcoco_dataset import *

# smoker数据集的类别
obj_list = ['person', 'cigarette']


class SMOKER_DET_Dataset(Dataset):
    def __init__(self, root_dir, set='trainval', transform=None, color_prob=0):
        # self.root_dir = root_dir
        self.data_dir = root_dir # datasets/smoker_det
        self.processed_dir = os.path.join(self.data_dir, "annotations")
        self.setname = set
        self.transform = transform
        self.color_prob = color_prob

        self.load_object_category()
        self.load_verb_category()
        self.load_hoi_category()
        self.load_ann_list()
        self.load_ann_by_image()

    def load_object_category(self):
        # 把人物个类别分别对应上标签序号,一一对应，整理成字典
        self.obj_to_id = {}
        self.id_to_obj = {}
        for id, obj in enumerate(obj_list):
            if obj != "":
                self.obj_to_id[obj] = id
                self.id_to_obj[id] = obj
        assert len(self.obj_to_id) == 2
        assert len(self.id_to_obj) == 2

    def load_verb_category(self):
        # 将动作和id加载出来
        self.id_to_verb = {}
        self.verb_to_id = {}
        verb_list_path = os.path.join(self.processed_dir, "verb_list.json")
        with open(verb_list_path, "r") as file:
            verb_list = json.load(file)
        for item in verb_list:
            # 每一个元素都是一个字典，包含id,name
            id = int(item["id"])
            name = item["name"]
            self.id_to_verb[id] = name
            self.verb_to_id[name] = id
        self.num_verbs = len(self.verb_to_id) # 包含的动作数量
       

    def load_hoi_category(self):
        # 将hoi的3个类的动作和物体一一映射起来
        self.hoi_to_objid = {}
        self.hoi_to_verbid = {}
        hoi_list_path = os.path.join(self.processed_dir, "hoi_list.json")
        with open(hoi_list_path, "r") as file:
            hoi_list = json.load(file)
        for item in hoi_list:
            # 每一个元素都是一个字典，包含id，object,verb
            hoi_id = int(item["id"])
            object = item["object"]
            object = object.replace("_", " ") # 如果object中有“_”的，需要替换成空格
            verb = item["verb"]
            self.hoi_to_objid[hoi_id] = self.obj_to_id[object] # 把hoi的3个id和物体的id匹配起来
            self.hoi_to_verbid[hoi_id] = self.verb_to_id[verb] # 把hoi的3个id和动作的id匹配起来
        self.num_hois = len(self.hoi_to_verbid)

    def load_ann_list(self):
        # 载入标注文件，这就是最重要的文件“anno_list.json”
        ann_list_path = os.path.join(self.processed_dir, "anno_list.json")
        with open(ann_list_path, "r") as file:
            ann_list = json.load(file)
            # print("dataset nums:", len(ann_list))
        # 将所有标注文件划分为训练集和测试集
        split_ann_list = []
        for item in ann_list:
            if self.setname in item["global_id"]:
                split_ann_list.append(item)
        self.split_ann_list = split_ann_list
       

    def load_ann_by_image(self):
        # 最为关键的一个函数，加载数据并进行处理
        self.ann_by_image = []
        self.hoi_count = np.zeros(self.num_hois).tolist() # 生成3个元素全0的列表，用于统计每个交互的数量
        self.verb_count = np.zeros(self.num_verbs).tolist() # 生成3个元素全0的列表，用于统计动作数量

        for image_id, image_item in enumerate(self.split_ann_list):
            img_anns = {}

            image_path_postfix = image_item["image_path_postfix"]
            img_path = os.path.join(self.data_dir, "images", image_path_postfix)
            img_anns["img_path"] = img_path # 获取完整的图片路径，加入到该图片的标注中

            hois = image_item["hois"] # 获取该图片的所有交互标注信息

            inters = []  # (human_bbox, object_bbox, object_category, [action_category])
            instances = []  # (instance_bbox, instance_category, [human_actions], [object_actions])

            for idx, hoi in enumerate(hois):
                id_to_inter = {}  # (human_id, object_id) : (human_bbox, object_bbox, object_category, [action_category])
                id_to_human = {}  # human_id: (instance_bbox, instance_category, [human_actions], [])
                id_to_object = {}  # object_id: (instance_bbox, instance_category, [object_actions])

                hoi_id = int(hoi["id"]) # 获取交互的动作类别id
                if hoi["invis"]:
                    # 如果物体完全看不到了，就跳过这个数据
                    continue
                for i in range(len(hoi["connections"])):
                    # 逐步处理每个交互
                    connection = hoi["connections"][i] # 获取该交互所关联的人的框和物的框
                    human_bbox = hoi["human_bboxes"][connection[0]] # 取人的框
                    object_bbox = hoi["object_bboxes"][connection[1]] # 取物的框
                    inter_id = tuple([idx] + connection) # idx表示第几个交互(0,0,0)
                    human_id = tuple([idx] + [connection[0]]) # (0,0)
                    object_id = tuple([idx] + [connection[1]]) # (0,0)

                    self.hoi_count[hoi_id - 1] += 1 # hoi_id是从1开始的
                    self.verb_count[self.hoi_to_verbid[hoi_id]-1] += 1 # ver_id是从1开始的

                    if inter_id in id_to_inter:
                        id_to_inter[inter_id][3].append(self.hoi_to_verbid[hoi_id])
                

                    else:
                        """ 这部分最重要，整合了人的框，物的框，物的类别，动作类别 """
                        item = []
                        item.append(human_bbox)
                        item.append(object_bbox)
                        item.append(self.hoi_to_objid[hoi_id])
                        item.append([self.hoi_to_verbid[hoi_id]])
                        id_to_inter[inter_id] = item

                    if human_id in id_to_human:
                        id_to_human[human_id][2].append(self.hoi_to_verbid[hoi_id])
                    else:
                        id_to_human[human_id] = [human_bbox, 0, [self.hoi_to_verbid[hoi_id]], []]

                    if object_id in id_to_object:
                        id_to_object[object_id][3].append(self.hoi_to_verbid[hoi_id])
                    else:
                        id_to_object[object_id] = [object_bbox, self.hoi_to_objid[hoi_id], [], [self.hoi_to_verbid[hoi_id]]]

                inters += list(id_to_inter.values())
                instances = instances + list(id_to_human.values()) + list(id_to_object.values())
            
            unique_instances = []
            for inst in instances:
                m = 0.7
                minst = None
                for uinst in unique_instances:
                    if inst[1] == uinst[1] and single_iou(inst[0], uinst[0]) > m:
                        minst = uinst
                        m = single_iou(inst[0], uinst[0])
                if minst is None:
                    unique_instances.append(inst)
                else:
                    minst[2] += inst[2]
                    minst[3] += inst[3]

            unique_inters = []
            for inter in inters:
                m = 0.7 ** 2
                minter = None
                for uinter in unique_inters:
                    hiou = single_iou(inter[0], uinter[0])
                    oiou = single_iou(inter[1], uinter[1])
                    if inter[2] == uinter[2] and hiou > 0.7 and oiou > 0.7 and hiou*oiou > m:
                        minter = uinter
                        m = hiou * oiou
                if minter is None:
                    unique_inters.append(inter)
                else:
                    minter[3] += inter[3]

            img_anns["interaction"] = unique_inters
            img_anns["instance"] = unique_instances
            self.ann_by_image.append(img_anns)
        self.num_images = len(self.ann_by_image)
        print("num_images:", self.num_images)
        with open("smoker-det_hoi_count.json", "w") as file:
            json.dump(self.hoi_count, file)
        with open("smoker-det_verb_count.json", "w") as file:
            json.dump(self.verb_count, file)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img_item = self.ann_by_image[index]
        img = self.load_img(img_item["img_path"])

        annot_bbox = {"instance": [], "interaction": []}
        for i, ann in enumerate(img_item["instance"]):
            tmp = np.zeros(4 + 1 + self.num_verbs * 2)  # (bbox, obj_cat, human action, object action)
            tmp[0:4] = ann[0]  # bbox
            tmp[4] = ann[1]  # object category
            human_act = np.zeros(self.num_verbs)  # human action
            obj_act = np.zeros(self.num_verbs)   # object action

            h_acts = np.array(ann[2]) - 1
            o_acts = np.array(ann[3]) - 1

            if h_acts.shape[0] > 0:
                human_act[h_acts] = 1
            if o_acts.shape[0] > 0:
                obj_act[o_acts] = 1

            tmp[5:5+self.num_verbs] = human_act
            tmp[5+self.num_verbs:5+2*self.num_verbs] = obj_act
            annot_bbox["instance"].append(tmp)

        for i, ann in enumerate(img_item["interaction"]):
            tmp = np.zeros(12 + 1 + self.num_verbs)  # (human bbox, object bbox, union bbox, obj category, union action)
            tmp[0:4] = ann[0]
            tmp[4:8] = ann[1]
            tmp[8:12] = self.merge_bbox(ann[0], ann[1])
            tmp[12] = ann[2]

            union_acts = np.zeros(self.num_verbs)

            u_acts = np.array(ann[3]) - 1
            union_acts[u_acts] = 1
            tmp[13:] = union_acts
            annot_bbox["interaction"].append(tmp)

        for key in annot_bbox:
            annot_bbox[key] = np.array(annot_bbox[key])

        sample = {'img': img, 'annot': annot_bbox}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def merge_bbox(self, b1, b2):
        if b1[0] < 0:
            return b2
        if b2[0] < 0:
            return b1
        return [min(b1[0], b2[0]), min(b1[1], b2[1]),
                max(b1[2], b2[2]), max(b1[3], b2[3])]

    def load_img(self, img_path):
        # 载入图片，用于训练前归一化颜色
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if np.random.uniform(0, 1) < self.color_prob:
            pil_img = Image.fromarray(img)
            img = np.array(randomColor(pil_img))
        return img.astype(np.float32) / 255.


def single_iou(a, b, need_area = False):
    # a(x1, y1, x2, y2)
    # b(x1, y1, x2, y2)

    area = (b[2] - b[0]) * (b[3] - b[1])
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    iw = max(iw, 0)
    ih = max(ih, 0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + area - iw * ih
    ua = max(ua, 1e-8)

    intersection = iw * ih
    IoU = intersection / ua
    if need_area:
        return IoU, intersection, ua
    else:
        return IoU

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    func = [color_enhance, brightness_enhance, contrast_enhance, sharpness_enchance]
    random.shuffle(func)
    for f in func:
        image = f(image)
    return image


def color_enhance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    return color_image


def brightness_enhance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    return brightness_image


def contrast_enhance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    contrast_image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    return contrast_image


def sharpness_enchance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    sharp_image = ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return sharp_image


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5, crop_prob=1):
        image, annots = sample['img'], sample['annot']
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            for key in ["instance", "interaction"]:
                if len(annots[key]) == 0:
                    continue
                if key == "instance":
                    t = 1
                else:
                    t = 3
                for i in range(t):
                    w = annots[key][:, 4*i+2] - annots[key][:, 4*i+0]

                    annots[key][:, 4*i+2][annots[key][:, 4*i+2] > 0] = (cols - annots[key][:, 4*i+0])[annots[key][:, 4*i+2] > 0]
                    annots[key][:, 4*i+0][annots[key][:, 4*i+0] > 0] = (annots[key][:, 4*i+2] - w)[annots[key][:, 4*i+0] > 0]

        if np.random.rand() < crop_prob:
            raw_h = image.shape[0]
            raw_w = image.shape[1]

            if len(annots["interaction"]) > 0:
                xmin = np.min(annots["interaction"][:, 8])
                ymin = np.min(annots["interaction"][:, 9])
                xmax = np.max(annots["interaction"][:, 10])
                ymax = np.max(annots["interaction"][:, 11])
            else:
                xmin = raw_w
                ymin = raw_h
                xmax = 0
                ymax = 0

            if len(annots["instance"]) > 0:
                instance_area = (annots["instance"][:, 2] - annots["instance"][:, 0]) * (annots["instance"][:, 3] - annots["instance"][:, 1])

            xmin = min(xmin, raw_w - raw_w / 2)
            ymin = min(ymin, raw_h - raw_h / 2)
            xmax = max(xmax, raw_w / 2)
            ymax = max(ymax, raw_h / 2)

            x1 = int(np.random.uniform(0, xmin))
            y1 = int(np.random.uniform(0, ymin))

            x2 = int(np.random.uniform(max(xmax, x1+raw_w/2), raw_w))
            y2 = int(np.random.uniform(max(ymax, y1+raw_h/2), raw_h))
            # x2 = int(np.random.uniform(xmax, raw_w)) + 1
            # y2 = int(np.random.uniform(ymax, raw_h)) + 1

            if len(annots["interaction"]) > 0:
                annots["interaction"][:, [0, 2, 4, 6, 8, 10]] -= x1
                annots["interaction"][:, [1, 3, 5, 7, 9, 11]] -= y1
            if len(annots["instance"]) > 0:
                annots["instance"][:, [0, 2]] -= x1
                annots["instance"][:, [1, 3]] -= y1
                new_instance_area = (annots["instance"][:, 2] - annots["instance"][:, 0]) * (annots["instance"][:, 3] - annots["instance"][:, 1])
                remain_idx = (new_instance_area / instance_area) > 0.5
                annots["instance"] = annots["instance"][remain_idx]

            image = image[y1:y2, x1:x2, :]

        sample = {'img': image, 'annot': annots}

        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        for key in ["instance", "interaction"]:
            if len(annots[key]) == 0:
                continue
            if key == "instance":
                t = 4
            else:
                t = 12
            annots[key][:, :t] *= scale
            annots[key][:, :t][annots[key][:, :t] < 0] = -1

        for key in ["instance", "interaction"]:
            annots[key] = torch.from_numpy(np.array(annots[key]))

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': annots, 'scale': scale}


def collater(data):
    imgs = [s['img'] for s in data]
    annots = {}
    annots["instance"] = [s['annot']['instance'] for s in data]
    annots["interaction"] = [s['annot']['interaction'] for s in data]

    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = {
        "instance": max(instance.shape[0] for instance in annots["instance"]),
        "interaction": max(interaction.shape[0] for interaction in annots["interaction"])
    }

    annot_len = {
        "instance": 0,
        "interaction": 0
    }
    # annot_len = {
    #     "instance": 4 + 1 + len(sub_label_to_class) + len(obj_label_to_class),
    #     "interaction": 12 + 1 + len(label_to_class)
    # }
    for key in ["instance", "interaction"]:
        for item in annots[key]:
            if len(item.shape) > 1:
                annot_len[key] = item.shape[1]
                break

    annot_padded = {}

    for key in ["instance", "interaction"]:
        if max_num_annots[key] > 0:
            annot_padded[key] = torch.ones((len(annots[key]), max_num_annots[key], annot_len[key])) * -1
            for idx, annot in enumerate(annots[key]):
                if annot.shape[0] > 0:
                    annot_padded[key][idx, :annot.shape[0], :] = annot
        else:
            annot_padded[key] = torch.ones((len(annots[key]), 1, annot_len[key])) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}
    # annot: {instance: Tensor, interaction: Tensor}
    # instance: n*m*(obj bpx,obj_cls,sub_act_cls_one_hot,obj_act_cls_one_hot),
    # interaction: n*m*(sub/obj/act box, obj_cls, act_cls_one_hot)
    # n: batch size, m: max count in single image



def draw_bbox(imgs, annots):
    batch_size = len(imgs)
    for i in range(batch_size):
        img_np = imgs[i].permute(1, 2, 0).numpy() * 255
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np = img_np.astype(np.uint8)

        img_inter = copy.deepcopy(img_np)
        for annot in annots["interaction"][i]:
            if annot[0] < 0:
                continue
            cv2.rectangle(img_inter, (annot[0], annot[1]), (annot[2], annot[3]), color=(255, 0, 0))
            cv2.rectangle(img_inter, (annot[4], annot[5]), (annot[6], annot[7]), color=(0, 255, 0))
            cv2.rectangle(img_inter, (annot[8], annot[9]), (annot[10], annot[11]), color=(0, 0, 255))
        cv2.imwrite("imgs/smoker_%d_inter.png"%i, img_inter)

        img_inst = copy.deepcopy(img_np)
        for annot in annots["instance"][i]:
            if annot[0] < 0:
                continue
            cv2.rectangle(img_inst, (annot[0], annot[1]), (annot[2], annot[3]), color=(255, 0, 0))

        cv2.imwrite("imgs/smoker_%d_inst.png" % i, img_inst)

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    training_set = SMOKER_DET_Dataset(root_dir="datasets/smoker_det", set="trainval",
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    training_params = {'batch_size': 3,
                       'shuffle': False,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': 0}
    training_generator = DataLoader(training_set, **training_params)


    print("iters = anno_list/batchsize:", len(training_generator))


