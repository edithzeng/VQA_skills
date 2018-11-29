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

def main():
    pass
if __name__ == "__main__":
    main()


# aggregte visual features for prediction
def api_visual_features(df, cv_key, logfile, test_data=False, error_rows=False):
    features = []
    file = open(logfile, "a")
    file.write("qid,adult,categories,descriptions,tags,txt,col\n")
    for i, row in df.iterrows():
        qid = row[0]
        if not test_data:
            txt, col, = row[5], row[6]
        if error_rows:
            captions, descriptions, tags, categories, adult = None, None, None, None, None
        else:
            image_url = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/%s'%qid
            captions, descriptions, tags, categories, adult = api_call(image_url, cv_key)
        # aggregate features and task class
        result = {'qid': qid,'adult': adult,'categories':categories,'descriptions':descriptions,'tags':tags}
        if not test_data:
            result['TXT'] = txt
            result['COL'] = col
            result_str = "{},{},{},{},{},{},{}\n".format(qid,adult,categories,descriptions,tags,txt,col)
        else:
            result_str = "{},{},{},{},{}\n".format(qid, adult, categories, descriptions, tags)
        file.write(result_str)
        features.append(result)
    file.close()
    return pd.DataFrame(features)


def nn_feature_extract(df, cv_key, logfile, test_data=False):
    features = []
    file = open(logfile, "a")
    file.write("qid,vgg_tags,alexnet_tags,txt,col\n")
    for i, row in df.iterrows():
        qid = row[0]
        if not test_data:
            txt, col = row[4], row[6]
        image_name = row[1]
        image_url = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/%s'%image_name
        vgg_tags = vgg_feature_extract(image_url)
        alexnet_tags = alexnet_feature_extract(image_url)
        if not test_data:
            result = {'qid': qid, 'vgg_tags': vgg_tags, 'alexnet_tags': alexnet_tags, 'txt': txt, 'col': col}
            result_str = "{},{},{},{},{}\n".format(qid, vgg_tags, alexnet_tags, txt, col)
        else:
            result = {'qid': qid, 'vgg_tags': vgg_tags, 'alexnet_tags': alexnet_tags}
            result_str = "{},{},{}\n".format(qid, vgg_tags, alexnet_tags)
        features.append(result)
        file.write(result_str)
    file.close()
    return pd.DataFrame(features)


# Examples from 
# https://gist.github.com/jbencook/9918217f866c1aa9967391ba62d123b5 and 
# https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
# https://pytorch.org/docs/stable/torchvision/models.html#id2
# VGG 13 (OxfordNet): K Simonyan and A Zisserman. https://arxiv.org/abs/1409.1556
# ResNet34: He et al. https://goo.gl/KHQcso.
def vgg_feature_extract(image_url):
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
    m = models.vgg13(pretrained=True)
    prediction = m(img)
    prediction = prediction.data.numpy().argmax()
    return labels[prediction]


def alexnet_feature_extract(image_url):
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
    m = models.alexnet(pretrained=True)
    prediction = m(img)
    prediction = prediction.data.numpy().argmax()
    return labels[prediction]


    
def api_call(image_url, cv_key):
    analysis = analyze_image(image_url, cv_key)
    captions = analysis['description']['captions']
    descriptions = None if len(captions)<1 else captions[0]['text']
    tags = None if len(analysis['description']['tags'])<1 else analysis['description']['tags']
    categories = None if len(analysis['categories'])<1 else analysis['categories'][0]['name']
    adult = analysis['adult']['isAdultContent']
    return captions, descriptions, tags, categories, adult

# Extract high level visual features from Microsoft cognitive service computer vision API
# Codes from in-class lab instructions
# https://notebooks.azure.com/anon-wnedpg/libraries/Fall2018-IntroToML/html/API_Calls.ipynb
def analyze_image(image_url, cv_key):
    vision_base_url = 'https://eastus.api.cognitive.microsoft.com/vision/v1.0'
    vision_analyze_url = vision_base_url + '/analyze?'
    headers = {'Ocp-Apim-Subscription-key': cv_key}
    params = {'visualfeatures': 'Adult,Categories,Description,Color,Faces,ImageType,Tags'}
    data = {'url': image_url}
    response = requests.post(vision_analyze_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    analysis = response.json()
    return analysis
