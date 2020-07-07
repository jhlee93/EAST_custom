import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset, detect, plot_boxes
import numpy as np
import shutil
from PIL import Image
import cv2
from binary_eval import confusion_matrix
import pandas as pd
from selective_search import selective_search, box_filter


def bbox_info(model_path, img_path, gt_path, scroe_thres=0.5, nms_thres=0.4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_list = os.listdir(img_path)
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')

    model = EAST().to(device) # to gpu
    print('\nCurrent Model Path: ', model_path)
    model.load_state_dict(torch.load(model_path))
    print('\nSuccess load model ...')
    
    model = model.eval()

    total_pred_boxes, total_gt_boxes = 0, 0
    predBoxes, imgWidth, imgHeight, gtBoxes = [], [], [], []

    # bbox info dataframe
    df = pd.DataFrame(columns=['img_name', 'inf_boxes', 'img_width', 'img_height', 'gt_boxes'])
    for i in image_list:
        img = Image.open(img_path + '/' + i)
        width, height = img.size

        imgWidth.append(int(width))
        imgHeight.append(int(height))
        
        # Prediction boxes
        pred_boxes = detect(img, model, device, score_threshold=scroe_thres, nms_threshold=nms_thres)

        # GT boxes
        anno = gt_path + '/' + i.split('.')[0] + '.txt'
        
        gt_boxes = []
        with open(anno, 'r') as f:
            while True:
                l = f.readline()
                if l == '':
                    break

                if '\n' in l:
                    l = l.split('\n')[0]
                
                gt_boxes.append([int(x) for x in l.split(',')[:8]])

        total_gt_boxes += len(gt_boxes)

        gtBoxes.append(len(gt_boxes))

        if pred_boxes is not None:
            total_pred_boxes += len(pred_boxes)
            print('InfBoxes: {}, GTBoxes: {}'.format(len(pred_boxes), len(gt_boxes)))

            predBoxes.append(len(pred_boxes))

        else: # Detection false
            predBoxes.append(0)
            print('InfBoxes: {}, GTBoxes: {}'.format(0, len(gt_boxes)))

    df['img_name'] = image_list
    df['inf_boxes'] = predBoxes
    df['img_width'] = imgWidth
    df['img_height'] = imgHeight
    df['gt_boxes'] = gtBoxes

    return df

def sel_search(img_path):
    image_list = os.listdir(img_path)
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')

    num_box_list, time_list = [], []
    
    total_images = len(image_list)
    for id, i in enumerate(image_list):
        print('selective search : [{} / {}]'.format(id, total_images))
        img = cv2.imread(img_path + '/' + i)
        img = cv2.resize(img, dsize=(500,500))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        s_start = time.time()
        boxes = selective_search(img, mode='single')

        num_box_list.append(len(boxes))
        print(len(boxes))
        time_list.append(time.time()-s_start)
    
    print('Total Time: {}'.format(sum(time_list)))
    
    return num_box_list
        
if __name__ == '__main__':
    model_name = './pths/east_vgg16.pth'
    img_path = os.path.abspath('./ENKR_DATA/train_img')
    gt_path = os.path.abspath('./ENKR_DATA/train_gt')
    score_threshold = 0.9
    nms_threshold = 0.4

    df = bbox_info(model_name, img_path, gt_path, score_threshold, nms_threshold)
    df.to_csv('./box_estimation/al_train.csv', index=False)

    """
    # Get Selective Search bbox info
    df = pd.read_csv('./box_estimation/al_train.csv')

    # df['selective_search'] = sel_search(img_path)
    df.to_csv('./box_estimation/al_train.csv', index=False)
    """