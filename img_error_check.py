import os
import cv2

base_path = './ICDAR_ENKR/test_img/'

img_list = os.listdir(base_path)
for t, i in enumerate(img_list):
    img_path = base_path+i

    img = cv2.imread(img_path)
    print(t, img.shape)
    cv2.imwrite(img_path, img)
    # try:
    #     image_shape = img.shape

    # except:
    #     os.remove(img_path)
    #     os.remove('./ICDAR_ENKR/train_gt/' + i.split('.')[0]+'.txt')
    #     print('Remove ==> ', i)

    # print('{}/{}'.format((l+1), total_image))
