"this module get HOLT-pred.json from test_results_final.pkl"
"add cigarette detector(post-precess) to detect object boxes w/o cigarette"

import os
import sys
import json
import pickle
import numpy as np
import numpy


from detect_yolox import detect

def get_image_name(anno_gt_json):
    "load test_gt.json and get all test image name."
    with open(anno_gt_json, "r") as file:
        anno_list = json.load(file)
    image_name = []
    for item in anno_list:
        image_name.append(item['file_name'])
    print("total test img:", len(image_name))
    return image_name


def compute_IOU(bbox1, bbox2):
    # computing area of each rectangles
    S_rec1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    S_rec2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(bbox1[1], bbox2[1])
    right_line = min(bbox1[3], bbox2[3])
    top_line = max(bbox1[0], bbox2[0])
    bottom_line = min(bbox1[2], bbox2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)



def trans_to_test_pred(anno_pred, image_name):
    "load test_results_final.pkl and transform to another test_pred style(use post-process)."
    with open(anno_pred, "rb") as file:
        test_pred_res = pickle.load(file)

    test_image_pred = []
    i = 1
    for img_id in test_pred_res.keys():
        img_name = image_name[img_id-1]
        img_path = os.path.join('../datasets/smoker_det/images/test', img_name)
        if len(test_pred_res[img_id]) != 0:
            best_img_id = 0
            for pred_i, pred in enumerate(test_pred_res[img_id]):
                if pred[2] != 2 or np.sum(pred[3]) == 0:
                    continue
                # post process: detect w/o cigarette
                [px1, py1, px2, py2] = [int(round(pred[1][0])), int(round(pred[1][1])), int(round(pred[1][2])), int(round(pred[1][3]))]
                smoke_box = detect(img_path)
                if smoke_box != []:
                    [x1, y1, x2, y2] = smoke_box
                    iou = compute_IOU([px1, py1, px2, py2], [x1, y1, x2, y2])
                    if iou == 0:
                        continue
                    else:
                        best_img_id = pred_i
                        break
                else:
                    continue

            pred = test_pred_res[img_id][best_img_id]
            img_id_pred = {}


            img_id_pred['file_name'] = image_name[img_id-1]
            
            img_id_pred['predictions'] = []

            human_pred = {}
            human_box = [int(round(pred[0][0])), int(round(pred[0][1])), int(round(pred[0][2])), int(round(pred[0][3]))] 
            human_pred['bbox'] = human_box
            human_pred['category_id'] = "1"
            human_pred['score'] = round(float(pred[4]), 3)
            img_id_pred['predictions'].append(human_pred)

            object_pred = {}
            object_box = [int(round(pred[1][0])), int(round(pred[1][1])), int(round(pred[1][2])), int(round(pred[1][3]))] 
            object_pred['bbox'] = object_box
            object_pred['category_id'] = str(pred[2])
            object_pred['score'] = round(float(pred[5]), 3)
            img_id_pred['predictions'].append(object_pred)

            img_id_pred['hoi_prediction'] = []
            hoi_id = np.argmax(pred[3]) + 1
    
            hoi = {}
            hoi["subject_id"] = 0
            hoi["object_id"] = 1
            hoi["category_id"] = hoi_id
            hoi_score = np.max(pred[3]) / np.sum(pred[3])
            hoi["score"] = round(float(hoi_score), 3)
            img_id_pred['hoi_prediction'].append(hoi)
        else:
            img_id_pred = {}
            img_id_pred['file_name'] = image_name[img_id-1]
            img_id_pred['predictions'] = []
            img_id_pred['hoi_prediction'] = []
        test_image_pred.append(img_id_pred)

        i += 1
    return test_image_pred

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    anno_gt_json = "HOLT_gt.json"
    pred_anno_file = "../logs/smoker-detection/test_results/test_bbox_results_final.pkl"
    test_pred_file = "HOLT-pred.json"
    
    image_name = get_image_name(anno_gt_json)
    test_pred = trans_to_test_pred(pred_anno_file, image_name)
    json.dump(test_pred, open(test_pred_file, 'w'), cls=NumpyEncoder)
