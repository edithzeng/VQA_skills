import pandas as pd
import os
import json
import pprint
import io
import requests
import time
import sys
import urllib.error as error
from pprint import pprint
import skimage
from PIL import Image
import http.client, urllib.request, urllib.parse, urllib.error, base64
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO

def analyze_image(vision_base_url, image_url, key):
    vision_analyze_url = vision_base_url + '/analyze?'
    headers = {'Ocp-Apim-Subscription-key': key}
    # params = {'visualfeatures': 'Categories,Description,Color,Faces,ImageType,Tags'}
    params = {'visualfeatures': 'Color,Tags'}
    data = {'url': image_url}
    response = requests.post(vision_analyze_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    analysis = response.json()
    return analysis

def write_to_file(vision_base_url, key, df, output_file_path, dataset): 
    if (not (isinstance(df,pd.DataFrame) and dataset in ['vizwiz','vqa'])):
        print("Check argv")
        raise ValueError
    file = open(output_file_path, 'w+')
    file.write("qid;question;tags;dominant_colors\n")
    local_image = False
    image_url_base = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/'
    if dataset == 'vqa':
        local_image = True
        image_url_base = '../../VQA_data/images/'
    n = 1
    for i, row in df.iterrows():
        if (n%100 == 0):
            print("{0:.0%}".format(float(n)/len(df)), flush=True)
        qid = row[0]
        img = row[1]
        qsn = row[2]
        image_url  = "{}{}".format(image_url_base,img)
        analysis = analyze_image(vision_base_url, image_url, key)
        tags = analysis['tags']
        dominant_colors = result['color']['dominantColors']
        result_str = "{};{};{};{}\n".format(qid,qsn,tags,dominant_colors)
        file.write(result_str)
        n += 1
    file.close()
    print("Color recognition results for {} written to {}".format(dataset, output_file_path))