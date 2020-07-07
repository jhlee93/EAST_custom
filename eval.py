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


def evaluation(model_path, test_img_path, test_gt_path, iou_thres, scroe_thres, nms_thres):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	valid_images = os.listdir(test_img_path)
	if '.DS_Store' in valid_images:
		valid_images.remove('.DS_Store')

	# Init.
	fTP, fFP, fFN, total_recall, total_precision, total_pred_boxes, total_gt_boxes, total_success_boxes = 0, 0, 0, 0, 0, 0, 0, 0
	pred_times = []

	model = EAST().to(device) # to gpu
	print('\nCurrent Model Path: ', model_path)
	model.load_state_dict(torch.load(model_path))
	print('\nSuccess load model ...')
	

	err_images = []
	diff_list = []

	model = model.eval()
	for i in valid_images:
		img = Image.open(test_img_path + '/' + i)

		pred_start = time.time()
		# Prediction boxes
		pred_boxes = detect(img, model, device, score_threshold=scroe_thres, nms_threshold=nms_thres)

		pred_end = time.time() - pred_start
		pred_times.append(pred_end)
		
		gt_boxes = []
		# GT boxes
		anno = test_gt_path + '/' + i.split('.')[0] + '.txt'
		
		with open(anno, 'r') as f:
			while True:
				l = f.readline()
				if l == '':
					break

				if '\n' in l:
					l = l.split('\n')[0]
				
				gt_boxes.append([int(x) for x in l.split(',')[:8]])

		# number of boxes
		n_gt_boxes = len(gt_boxes)
		n_pred_boxes = 0

		if pred_boxes is not None:
			n_pred_boxes = len(pred_boxes)

			total_pred_boxes += n_pred_boxes
			total_gt_boxes += n_gt_boxes
			diff_boxes = n_gt_boxes - n_pred_boxes
			diff_list.append(diff_boxes)

			# Current Image Acc
			TP, FP, FN, r, p, thresh_bboxes = confusion_matrix(pred_boxes, gt_boxes, iou_thres)
			total_success_boxes += len(thresh_bboxes)

			# Draw pred boxes
			plot_img = plot_boxes(img, thresh_bboxes, False)
			plot_img = plot_boxes(img, gt_boxes, True)
			plot_img = plot_img.convert('RGB')
			plot_img = np.array(plot_img)
			plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
			plot_img = cv2.putText(plot_img, 'Inf Box : '+str(len(thresh_bboxes)), (10,50),
									cv2.FONT_HERSHEY_COMPLEX_SMALL, 3., (0,0,255), 2)# plot prediction
			plot_img = cv2.putText(plot_img, 'GT Box : '+str(len(gt_boxes)), (int(plot_img.shape[1]/2),50),
									cv2.FONT_HERSHEY_COMPLEX_SMALL, 3., (255,0,0), 2)# plot gt
			cv2.imwrite('./test_output/' + i, plot_img)

			if r == 0.:
				err_images.append(i)

			# print('Image: {}  Recall: {:.4f}  Precision: {:.4f}  PredBoxes: {}  GTBox: {}  PredTime: {:.4f}sec'.format(
			# 	i, r, p, len(pred_boxes), len(gt_boxes), pred_end))

		else: # Detection false
			TP, FP = 0, 0
			FN = n_gt_boxes
			diff_list.append(n_gt_boxes)


			# print('Image: {},  Recall: {:.4f}  Precision: {:.4f}  PredBoxes: {}  GTBox: {}  PredTime: {:.4f}sec'.format(
			# 		i, 0, 0, 0, len(gt_boxes), pred_end))

		# Total confusion matrix
		fTP += TP
		fFP += FP
		fFN += FN

	if fTP != 0:
		total_recall = fTP / total_gt_boxes
		total_precision = fTP / total_pred_boxes

	print("total gt boxes: {}, total pred boxes: {}, success boxes: {}".format(total_gt_boxes, total_pred_boxes, total_success_boxes))
	# print('IoU: {}, ConfScore: {}, TP: {}, FN: {}, FP: {}, avgTP: {:.2f}, avgFN: {:.2f}, avgFP: {:.2f}'.format(
	# 	iou_thres, scroe_thres,
	# 	fTP, fFN, fFP,
	# 	(fTP / 100), (fFN / 100), (fFP / 100)
	# ))

	print('='*50)
	print('TestImage: {}  Recall: {:.4f}  Precision: {:.4f}  AvgPredTime: {:.4f}sec'.format(
		len(valid_images),
		total_recall,
		total_precision,
		sum(pred_times)/len(pred_times)
	))
	print()

	# Calc RMSE
	min_diff_boxes = min(diff_list)
	max_diff_boxes = max(diff_list)
	m = 0
	for i in diff_list:
		m += (i ** 2)
	
	mse = m / len(diff_list)
	rmse = mse ** 0.5

	print('Higher: {}, Lower: {}, MSE: {}, RMSE: {}'.format(min_diff_boxes, max_diff_boxes, mse, rmse))

	return total_recall, total_precision
	


if __name__ == '__main__': 
	model_name = './pths/ENKR_epoch_198.pth'
	test_img_path = os.path.abspath('./ENKR_DATA/test_img')
	test_gt_path = os.path.abspath('./ENKR_DATA/test_gt')

	# recall, precision = evaluation(
	# 	model_name,
	# 	test_img_path,
	# 	test_gt_path,
	# 	iou_thres=0.5,
	# 	scroe_thres=0.9,
	# 	nms_thres=0.4)

	# Eval
	# conf_score = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
	conf_score = [0.5, 0.7]
	# iou_list = [x/10 for x in range(3, 9)]
	iou_list = [0.5]

	for score in conf_score:
		scores, ious, rc, pc = [], [], [], []

		for i in iou_list:
			if os.path.isfile('./pths/optimize2.csv'):
				optimize_df = pd.read_csv('./pths/optimize2.csv')

				scores = optimize_df['conf_score'].to_list()
				ious = optimize_df['iou'].to_list()
				rc = optimize_df['recall'].to_list()
				pc = optimize_df['precision'].to_list()

			optimize_df = pd.DataFrame(columns=['conf_score', 'iou', 'recall', 'precision'])

			recall, precision = evaluation(
									model_name,
									test_img_path,
									test_gt_path,
									iou_thres=i,
									scroe_thres=score,
									nms_thres=0.4)

			scores.append(score)
			ious.append(i)
			rc.append(recall)
			pc.append(precision)

			optimize_df['conf_score'] = scores
			optimize_df['iou'] = ious
			optimize_df['recall'] = rc
			optimize_df['precision'] = pc

			optimize_df.to_csv('./pths/optimize2.csv', index=False)
