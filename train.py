import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np
from binary_eval import confusion_matrix
from detect import detect
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, test_img_path, test_gt_path):
	# Validation images
	valid_images = os.listdir(test_img_path)

	file_num = len(os.listdir(train_img_path))

	# Train loader
	trainset = custom_dataset(train_img_path, train_gt_path, 0.25, 512)
	train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True)

	# Valid loader
	testset = custom_dataset(test_img_path, test_gt_path)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	model = model.to(device)
	model.load_state_dict(torch.load('./pths/ENKR_epoch_200.pth')) # pretrained
	model = model.eval()
	

	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)
	
	# Log
	epoch_list, eloss_list, recall_list, precision_list, epoch_time_list = [], [], [], [], []


	for epoch in range(epoch_iter):
		epoch += 200


		model = model.train()
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))

		epoch_end_time = time.time() - epoch_time
		# print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'ENKR_epoch_{}.pth'.format(epoch+1)))

		# Evaluation
		fTP, fFP, fFN = 0, 0, 0
		recall, precision = 0, 0
		total_pred_boxes = 0

		
		if (epoch+1) >= 0:
			model = model.eval()
			for i in valid_images:
				img = Image.open(test_img_path + '/' + i)
				# Prediction boxes
				
				pred_boxes = detect(img, model, device, score_threshold=0.7, nms_threshold=0.2)
				
				if pred_boxes is not None:
					total_pred_boxes += len(pred_boxes)

					# GT boxes
					anno = test_gt_path + '/' + i.split('.')[0] + '.txt'
					with open(anno, 'r') as f:
						gt_boxes = []
						while True:
							l = f.readline()
							if l == '':
								break

							if '\n' in l:
								l = l.split('\n')[0]
							
							gt_boxes.append([int(x) for x in l.split(',')[:8]])

				
						TP, FP, FN, _, _ = confusion_matrix(pred_boxes, gt_boxes, 0.5)
				else: # Detection false
					TP, FP = 0, 0
					FN = len(gt_boxes)

				# Total confusion matrix
				fTP += TP
				fFP += FP
				fFN += FN

			if fTP != 0:
				recall = fTP / total_pred_boxes
				precision = fTP / (fTP + fFP)

		# Each epoch log save
		eloss = epoch_loss/int(file_num/batch_size)

		if os.path.isfile('./pths/train_log.csv'):
			train_log = pd.read_csv('./pths/train_log.csv')
			epoch_list = train_log['epoch'].to_list()
			eloss_list = train_log['loss'].to_list()
			recall_list = train_log['recall'].to_list()
			precision_list = train_log['precision'].to_list()
			epoch_time_list = train_log['epoch_time'].to_list()

		
		train_log = pd.DataFrame(columns=['epoch', 'loss', 'recall', 'precision', 'epoch_time'])
		
		epoch_list.append(epoch+1)
		eloss_list.append(eloss)
		recall_list.append(recall)
		precision_list.append(precision)
		epoch_time_list.append(epoch_end_time)
		# log
		train_log['epoch'] = epoch_list
		train_log['loss'] = eloss_list
		train_log['recall'] = recall_list
		train_log['precision'] = precision_list
		train_log['epoch_time'] = epoch_time_list
		train_log.to_csv('./pths/train_log.csv', index=False)

		# log visualization
		plt.figure(figsize=(15,8))
		plt.plot(train_log['epoch'], train_log['loss'], c='blue', label='loss')
		plt.title('Epoch Loss')
		plt.xlabel('epoch')
		plt.savefig('./pths/epoch_loss.png')

		plt.figure(figsize=(15,8))
		plt.plot(train_log['epoch'], train_log['recall'], c='red', label='recall')
		plt.plot(train_log['epoch'], train_log['precision'], c='blue', label='precision')
		plt.title('Accuracy')
		plt.xlabel('epoch')
		plt.legend()
		plt.savefig('./pths/accuracy.png')
		
		print('='*50)
		print('Epoch: {}, Loss: {}, Recall: {}, Precision: {}, EpochTime: {}'.format(
			(epoch+1),
			eloss,
			recall,
			precision,
			epoch_end_time
		))
		print()


if __name__ == '__main__':
	train_img_path = os.path.abspath('./ENKR_DATA/train_img')
	train_gt_path  = os.path.abspath('./ENKR_DATA/train_gt')
	valid_img_path = os.path.abspath('./ENKR_DATA/test_img')
	valid_gt_path = os.path.abspath('./ENKR_DATA/test_gt')

	pths_path      = './pths'
	batch_size     = 24
	lr             = 1e-3
	num_workers    = 4
	epoch_iter     = 200
	save_interval  = 2
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, valid_img_path, valid_gt_path)