import os
import shutil
import json

def main():
    image_list = os.listdir('./train_img/')
    gt_list = os.listdir('./train_gt/')

    for i in image_list:
        if i.split('.')[0] + '.txt' not in gt_list:
    
            if i.split('.')[1] == 'gif':
                print(i)

main()