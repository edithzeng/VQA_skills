import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import pprint
from pprint import pprint
import requests
from skimage import io
import cv2
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def main():
    pass
if __name__ == "__main__":
    main()


# aggregte visual features for prediction
def aggregate_visual_features(df_train, df_train_features, cv_key):
    existing_features = df_train_features.copy()[['QID','I_NUM_CATEGORIES',
                                                  'I_NUM_TAGS','I_NUM_COLOURS',
                                                  'I_NUM_FACES']]
    features = []
    for i, row in df_train.iterrows():
        # use existing features from Nilavra
        qid, txt, obj, col, cnt, oth = row[0], row[4], row[5], row[6], row[7], row[8]
        df = existing_features.loc[existing_features['QID'] == qid]
        num_categories = df['I_NUM_CATEGORIES']
        num_tags = df['I_NUM_TAGS']
        num_colors = df['I_NUM_COLOURS']
        num_faces = df['I_NUM_FACES']
        # get high-level visual features with Microsoft cognitive service CV API
        image_name = row[1]
        image_url = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/%s'%image_name
        image_path = '../data/Images/%s'%image_name
        
        captions, descriptions, tags, categories, adult = api_call(image_url, cv_key)
        resnet_tags = resnet_feature_extract(image_path)
        vgg_tags = vgg_feature_extract(image_path)
        
        # aggregate features and class
        temp = {'num_categories':num_categories, 'num_tags': num_tags,
                'num_colors': num_colors, 'num_faces': num_faces, 
                'adult': adult, 'categories':categories, 
                'descriptions':descriptions, 'tags':tags,
                'resnet_tags':resnet_tags, 'vgg_tags':vgg_tags,
                'TXT': txt, 'OBJ': obj, 'CNT':cnt, 'OTH':oth}
        features.append(temp)
        
    return pd.DataFrame(features)

# extract visual feature using ResNet50 trained on ImageNet 
# https://keras.io/applications/#usage-examples-for-image-classification-models
# ResNet50 (pre-trained on ImageNet)
# https://arxiv.org/abs/1512.03385
def resnet_feature_extract(image_path):
    model = keras.applications.resnet50.ResNet50(weights='imagenet')
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    preds_list = decode_predictions(preds, top=3)[0]
    return [preds_list[i][1] for i in range(len(preds_list))]


# Oxford Net / VCG 16 (pre-trained on ImageNet)
# https://arxiv.org/abs/1409.1556
def vgg_feature_extract(image_path):
    model = VGG16(weights='imagenet')
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    preds_list = decode_predictions(preds, top=3)[0]
    return [preds_list[i][1] for i in range(len(preds_list))]
    
    
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
        image = io.imread(image_url)
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

