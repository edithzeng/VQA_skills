import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pprint
import io
import requests
import cv2

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
def aggregate_visual_features(df_train, cv_key):
    features = []
    for i, row in df_train.iterrows():
        # use existing features from Nilavra
        qid, txt, obj, col, cnt, oth = row[0], row[4], row[5], row[6], row[7], row[8]
        # get objects in the image with Microsoft cognitive service CV API
        image_name = row[1]
        image_url = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/%s'%image_name
        image_path = '../data/Images/%s'%image_name
        # denoise & get objects in the image with ResNet and VGG trained on ImageNet
        captions, descriptions, tags, categories, adult = api_call(image_url, cv_key)
        vgg_tags = cnn_feature_extract('vgg13', image_url)
        resnet_tags = cnn_feature_extract('resnet', image_url)
        # aggregate features and class
        temp = {'adult': adult, 'categories':categories, 
                'descriptions':descriptions, 'tags':tags,
                'vgg_tags': vgg_tags,
                'resnet_tags':resnet_tags,
                'TXT': txt, 'OBJ': obj, 'CNT':cnt, 'OTH':oth}
        features.append(temp)
    return pd.DataFrame(features)



# https://gist.github.com/jbencook/9918217f866c1aa9967391ba62d123b5
# https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
# https://pytorch.org/docs/stable/torchvision/models.html#id2
# VGG 13 (OxfordNet)
# ResNet34 
def cnn_feature_extract(model, image_url):
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
    m = models.vgg13(pretrained=True) if model == 'vgg13' else models.resnet34(pretrained=True)
    prediction = m(img)
    prediction = prediction.data.numpy().argmax()
    return labels[prediction]



    
def api_call(image_url, cv_key):
    try:
        ans = analyze_image(image_url, cv_key)
        captions = analysis['description']['captions']
        descriptions = None if len(captions)<1 else captions[0]['text']
        tags = None if len(analysis['description']['tags'])<1 else analysis['description']['tags']
        categories = None if len(analysis['categories'])<1 else analysis['categories'][0]['name']
        adult = analysis['adult']['isAdultContent']
    except Exception:
        captions, descriptions, tags, categories, adult = None, None, None, None, None
    return captions, descriptions, tags, categories, adult


# extract high level visual features from Microsoft cognitive service computer vision API
# codes from in-class lab instructions
def analyze_image(image_url, cv_key, visualize=False):
    vision_base_url = 'https://westcentralus.api.cognitive.microsoft.com/vision/v1.0'
    vision_analyze_url = vision_base_url + '/analyze?'
    if visualize:
        image = skimage.io.imread(image_url)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    # API params
    headers = {'Ocp-Apim-Subscription-key': cv_key}
    params = {'visualfeatures': 'Adult,Categories,Description,Color,Faces,ImageType,Tags'}
    data = {'url': image_url}
    # send request and get API response
    response = requests.post(vision_analyze_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    analysis = response.json()
    return analysis

