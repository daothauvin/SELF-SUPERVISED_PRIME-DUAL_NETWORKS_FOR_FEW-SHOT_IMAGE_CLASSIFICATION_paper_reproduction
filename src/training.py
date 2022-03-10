"""#Test"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import os
import numpy as np
from PIL import ImageEnhance
from PIL import Image
import json
from abc import abstractmethod
import torchvision.transforms as transforms
from pathlib import Path
from SSL import *

image_size = 224
base_file = "save_dir/base.json"
eval_file = "save_dir/val.json"
novel_file = "save_dir/novel.json"
n_shot = 1
n_way = 5
n_query = max(1, int(16* n_way/n_way))
nb_epoch = 300
lr = 0.01
alpha = 0.5

train_few_shot_params  = dict(n_way = n_way, n_support = n_shot) # n_support = n_shot
base_datamgr  = MyDataManager( image_size, n_query = n_query,  **train_few_shot_params)
val_datamgr  = MyDataManager( image_size, n_query = n_query,  **train_few_shot_params)

base_loader  = base_datamgr.get_data_loader(base_file , aug = True)
val_loader =  val_datamgr.get_data_loader(eval_file , aug = True)

best_savepath = Path("saved_models/best_model.pch")

backbone = Backbone(type = 10)
model = Model(n_way, n_shot, n_query, 512, backbone = backbone)
#model.load_state_dict(torch.load(Path("saved_models/model_dual_3.pch")))
#if savepath.is_file():
#	with savepath.open("rb") as fp:
#        	print("Loading saved model")
#       	model.load_state_dict(torch.load(savepath))


if torch.cuda.is_available():
	print("Using GPU")
	model = model.cuda()

for i in range(10):
	print("Iter {:d}".format(i))	
	
	print("Training prime")
	train(model, base_loader, val_loader, best_savepath, nb_epoch, lr, alpha = alpha, mode = "Prime")
	
	with best_savepath.open("rb") as fp:
		print("loaded best model")
		model.load_state_dict(torch.load(best_savepath))

	
	savepath = Path("saved_models/model_prime_"+str(i)+".pch")	
	with savepath.open("wb") as fp:
		print("Saving model prime", i)
		torch.save(model.state_dict(),fp)

	
	
	print("Training dual")		
	train(model, base_loader, val_loader, best_savepath, nb_epoch, lr, alpha = alpha, mode = "Dual")
	
	
	
	
	with best_savepath.open("rb") as fp:
		print("loaded best model")
		model.load_state_dict(torch.load(fp))
	 
	savepath = Path("saved_models/model_dual_"+str(i)+".pch")
	with savepath.open("wb") as fp:
		print("Saving model dual", i)
		torch.save(model.state_dict(),fp)
		
	print("On base")
	test(model, base_loader)
	print("On val")
	test(model, val_loader)
	print("Iteration ended")
	
print("training DONE!")









