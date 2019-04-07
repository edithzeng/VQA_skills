""" ResNet trained on ImageNet to extract image feature """

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import resnet152
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
from sklearn.metrics import roc_auc_score

def nn_feature_extract_vizwiz(df, logfile):
    features = []
    file = open(logfile, "a")
    file.write("qid,image_feature_vector\n")
    for i, row in df.iterrows():
        qid = row[0]
        image_name = row[1]
        image_url = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/%s'%image_name
        resnet_feature_vector = resnet_feature_extract(image_url=image_url)
        result = {'qid': qid, 'image_feature_vector': resnet_feature_vector}
        result_str = "{},{}\n".format(qid, resnet_feature_vector)
        features.append(result)
        file.write(result_str)
    file.close()
    return pd.DataFrame(features)

def nn_feature_extract_vqa(df, log_file):
	features = []
	file = open(logfile, "a")
	file.write("qid,image_feature_vector\n")
    for i, row in df.iterrows():
        qid = row[0]
        image_name = row[1]
        resnet_feature_vector = resnet_feature_extract(image_path=image_name)
        result = {'qid': qid, 'image_feature_vector': resnet_feature_vector}
        result_str = "{},{}\n".format(qid, resnet_feature_vector)
        features.append(result)
        file.write(result_str)
    file.close()
    return pd.DataFrame(features)

# ResNet34: He et al. https://goo.gl/KHQcso
def resnet_feature_extract(image_url=None, image_path=None):
	if image_url:
    	labels_url = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    	response = requests.get(labels_url)
    	labels = {int(key): value for key, value in response.json().items()}
    	response = requests.get(image_url)
    	img = Image.open(io.BytesIO(response.content))
    if image_path:
    	img = image.load_image(image_path)
    transform_pipeline = transforms.Compose([transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    img = Variable(img)
    model = models.ResNet152(weights='imagenet')
    features = model.predict(img)
    return features





# training 
#vizwiz_train = pd.read_csv("../../vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vqa_train = pd.read_csv('../../vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')

# validation
#vizwiz_val = pd.read_csv("../../vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')
vqa_val = pd.read_csv('../../vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')

# label training sets
#nn_feature_extract_vizwiz(vizwiz_train, "vizwiz_image_feature_train.csv")
nn_feature_extract_vqa(vqa_train, "vqa_image_feature_train.csv")

# label validation sets
#nn_feature_extract_vizwiz(vizwiz_train, "vizwiz_image_feature_val.csv")
nn_feature_extract_vqa(vqa_train, "vqa_image_feature_val.csv")