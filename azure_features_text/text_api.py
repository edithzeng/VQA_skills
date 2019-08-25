import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, copy, sys
import json
import pprint
import io
import requests
import cv2
import time
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
from sklearn.metrics import roc_auc_score

def df_cutoff(df):
    """ separate large dataframes to chunks """
    """ returns a list of pd.DataFrame """
    """ Azure limit up to 1000 records per request """
    buckets = len(df) // 1000
    if buckets > 0:
        chunks = [df.iloc[1000*i:min(1000*(i+1),len(df)),:] for i in range(buckets)]
    else:
        chunks = [copy.deepcopy(df)]
    return chunks

# querying text analytics API
def extract_keyphrases(documents, key, url):
    """ calls API and return key phrases """
    headers = {"Ocp-Apim-Subscription-Key": key}
    response = requests.post(url, headers=headers, json=documents)
    languages = response.json()
    try:
        return languages
    except KeyError:
        print("error index:", index, "error question:", question)
        print(languages)
        return
    
def process_questions(df):
    """ convert df to json """
    arr = []
    for i, row in df.iterrows():
        curr = {}
        curr["language"] = "en"
        curr["id"] = row[df.columns.get_loc("Unnamed: 0")]  # VizWiz QID is not numeric
        curr["text"] = row[df.columns.get_loc("QSN")]
        curr["OBJ"] = row[df.columns.get_loc("OBJ")]
        curr["TXT"] = row[df.columns.get_loc("TXT")]
        curr["COL"] = row[df.columns.get_loc("COL")]
        curr["CNT"] = row[df.columns.get_loc("CNT")]
        curr["OTH"] = row[df.columns.get_loc("OTH")]
        arr.append(curr)
    documents = {'documents': arr}
    return documents

def get_azure_keyphrases(df, filename, key, url):
    """ does not apply to test data w/o skill labels """
    # preprocess to divide large dataframe
    chunks = df_cutoff(df)
    # file IO
    file = open(filename, "w+")
    file.write("qid,question,obj,txt,col,cnt,oth\n")
    for c in chunks:
        # convert to json
        doc = process_questions(c)
        # get key phrases with api call
        result = extract_keyphrases(doc, key, url)
        # join with skill labels and write to csv
        for row in result['documents']:
            key_phrases = row['keyPhrases']
            record = c.loc[c['Unnamed: 0'] == int(row['id'])]
            qid, question = record.QID.item(), record.QSN.item()
            obj, oth = record.OBJ.item(), record.OTH.item()
            txt, col, cnt = record.TXT.item(), record.COL.item(), record.CNT.item()
            result_str = "{},{},{},{},{},{},{},{}\n".format(qid,question,key_phrases,obj,txt,col,cnt,oth)
            file.write(result_str)
    file.close()
    print("Complete - key phrases features written to", filename)
