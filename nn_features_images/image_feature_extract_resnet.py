""" ResNet 50 trained on ImageNet to extract image feature """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pprint
import io
import requests
import cv2
import time
import sys
import urllib.error as error
from pprint import pprint
import skimage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
import PIL
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics import roc_auc_score

def nn_feature_extract_vizwiz(df, logfile):
	features = []
	file = open(logfile, "a")
	file.write("qid,image_feature_vector\n")
	for i, row in df.iterrows():
		qid = row[0]
		image_name = row[1]
		image_url = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/%s'%image_name
		nn_feature_vector = feature_extract(image_url=image_url)
		result = {'qid': qid, 'image_feature_vector': nn_feature_vector}
		result_str = "{},{}\n".format(qid, nn_feature_vector)
		features.append(result)
		file.write(result_str)
	file.close()
	return pd.DataFrame(features)

def nn_feature_extract_vqa(df, logfile):
	features = []
	file = open(logfile, "a")
	file.write("qid,image_feature_vector\n")
	for i, row in df.iterrows():
		qid = row[0]
		image_name = row[1]
		image_path = "../../VQA_data/images/{}".format(image_name)
		nn_feature_vector = feature_extract(image_path=image_path)
		result = {'qid': qid, 'image_feature_vector': nn_feature_vector}
		result_str = "{},{}\n".format(qid, nn_feature_vector)
		features.append(result)
		file.write(result_str)
	file.close()
	return pd.DataFrame(features)

# ResNet34: He et al. https://goo.gl/KHQcso
# InceptionV3: https://arxiv.org/abs/1512.00567
# https://seba-1511.github.io/tutorials/beginner/data_loading_tutorial.html
def feature_extract(image_url=None, image_path=None):
	if image_url:
		labels_url = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
		response = requests.get(labels_url)
		labels = {int(key): value for key, value in response.json().items()}
		response = requests.get(image_url)
		img = Image.open(io.BytesIO(response.content))
		transform_pipeline = transforms.Compose([transforms.Resize(224),
											 transforms.CenterCrop(224),
											 transforms.ToTensor(),
											 transforms.Normalize(mean=[0.485, 0.456, 0.406],
																  std=[0.229, 0.224, 0.225])])
		img = transform_pipeline(img)
		img = img.unsqueeze(0)
		img = Variable(img)
		img = img.data.numpy()
		img = np.transpose(img, (0,2,3,1))
	if image_path:
		img = image.load_img(image_path, target_size=(224,224))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)
	model = ResNet50(weights='imagenet')
	features = model.predict(img)
	return features



# extract features for VQA training and validation data 
#vqa_train = pd.read_csv('../../vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vqa_val = pd.read_csv('../../vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
#nn_feature_extract_vqa(vqa_train, "vqa_image_feature_train.csv")
#print("finished VQA train")
nn_feature_extract_vqa(vqa_val, "vqa_image_feature_val.csv")
print("finished VQA val")

# extract image features for VizWiz training and validation data
#vizwiz_train = pd.read_csv("../../vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
#vizwiz_val = pd.read_csv("../../vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')
#nn_feature_extract_vizwiz(vizwiz_train, "vizwiz_image_feature_train.csv")
#nn_feature_extract_vizwiz(vizwiz_train, "vizwiz_image_feature_val.csv")
