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
from sklearn.feature_extraction.text import CountVectorizer
import nltk
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

def main():
    pass
if __name__ == "__main__":
    main()


# aggregte visual features for prediction
def aggregate_visual_features(cv_key):
    existing_features = df_train_features.copy()[['QID','I_NUM_CATEGORIES','I_NUM_TAGS','I_NUM_COLOURS','I_NUM_FACES']]
    features = []
    for i, row in df_train.iterrows():
        # use existing features from Nilavra
        qid, txt, obj, col, cnt, oth = row[0], row[4], row[5], row[6], row[7], row[8]
        df = existing_features.loc[existing_features['QID'] == qid]
        num_categories = df['I_NUM_CATEGORIES']
        num_tags = df['I_NUM_TAGS']
        num_colors = df['I_NUM_COLOURS']
        num_faces = df['I_NUM_FACES']
        # get high-level visual features with Microsoft cognitive service computer vision API
        image_name = row[1]
        image_url = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/%s'%image_name
        analysis = analyze_image(image_url, cv_key)
        if len(analysis['description']['captions']) > 0:
            descriptions = analysis['description']['captions'][0]['text']
        else:
            descriptions = None
        tags = None if len(analysis['description']['tags']) < 1 else analysis['description']['tags']
        categories = None if len(analysis['categories']) < 1 else analysis['categories'][0]['name']
        adult = analysis['adult']['isAdultContent']
        # aggregate features and class
        temp = {'num_categories':num_categories, 'num_tags': num_tags,
                'num_colors': num_colors, 'num_faces': num_faces, 
                'adult': adult, 'categories':categories, 
                'descriptions':descriptions, 'tags':tags,
                'TXT': txt, 'OBJ': obj, 'CNT':cnt, 'OTH':oth}
        features.append(temp)
    return pd.DataFrame(features)



# extract high level visual features from Microsoft cognitive service computer vision API
# codes from in-class lab instructions
def analyze_image(image_url, visualize=False, cv_key):
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

