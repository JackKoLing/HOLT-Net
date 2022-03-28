# HOLT-Net: Detecting Smokers via Human-Object Interaction with Lite Transformer Network


## Requirements

The code was trained with python 3.6,  pytorch 1.10.1, torchvision 0.11.2, CUDA 10.1, opencv-python 4.5.1, self-attention-cv 1.2.3 and Ubuntu 18.04.


## Getting Started

0. [Optional but recommended] create a new conda environment

   ```
   conda create -n HOLT-Net python=3.6
   ```

   And activate the environment
   
   ```
   conda activate HOLT-Net
   ```

1. Clone this repository:

   ```
   git clone https://github.com/JackKoLing/HOLT-Net.git
   ```

2. Install necessary packages (other common packages installed if need):

   ```
   pip install torch==1.10.1 torchvision==0.11.2 opencv-python==4.5.1 self-attention-cv==1.2.3 tqdm numpy tensorboard tensorboardX pyyaml
   ```


## Data Preparation (SCAU-SD Dataset)

Download [SCAU-SD]() dataset. We transform the annotations of SCAU-SD dataset to JSON format following [no_frills_hoi_det](https://github.com/BigRedT/no_frills_hoi_det).

We count the training sample number of each category in smoker-det_hoi_count.json and smoker-det_verb_count.json following [DIRV](https://github.com/MVIG-SJTU/DIRV). It serves as a weight when calculating loss.

### Dataset Structure:

Make sure to put the files in the following structure:

```
|-- datasets
|   |-- smoker_det
|	|	|-- images
|	| 	|	|-- trainval
|	|	|	|-- test
|	|	|-- annotations
|	| 	|	|-- anno_list.json
|	|	|	|-- hoi_list.json
|	|	|	|-- object_list.json
|	|	|	|-- smoker-det_hoi_count.json
|	|	|	|-- smoker-det_verb_count.json
|	|	|	|-- verb_list.json
```


## Pre-trained Weights

Download the pre-trained [model]() for HOI trained on [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), and the post-refinement model trained on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). Make sure to put them in `weights/` folder.


## Training

```
python train_smoker.py -c 1 --batch_size 8 --optim adamw --load_weights weights/efficientdet-d1_pretrained.pth
```

You may also adjust the saving directory and GPU number in `projects/smoker_det.yaml` if you have multi-GPUs.


## Test 

```
python test_smoker.py -c 1 -w $path to the checkpoint$
```

## Eval

```
cd eval
python get_test_pred_yolox.py
python eval_smker.py
```


## Acknowledge
The code is developed based on the architecture of [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), [DIRV](https://github.com/MVIG-SJTU/DIRV) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). We sincerely thank the authors for the excellent works!
