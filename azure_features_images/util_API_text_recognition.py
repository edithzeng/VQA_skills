""" API functions for images with text recognition labels """

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
from PIL import Image
from torch.autograd import Variable
import http.client, urllib.request, urllib.parse, urllib.error, base64
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO

# OCR - https://goo.gl/sR74GW
def ocr_request(vision_base_url, key, image_url, local_image=False):
    """ return a list of lowercase strings, may contain special characters """
    ocr_url = vision_base_url + "ocr"
    params  = {'language': 'en', 'detectOrientation': 'true'}
    if local_image:
        image_path = image_url
        image_data = open(image_path, "rb").read()
        headers    = {'Ocp-Apim-Subscription-Key': key,
                      'Content-Type': 'application/octet-stream'}
        response   = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    else:
        data       = {'url': image_url}
        headers    = {'Ocp-Apim-Subscription-Key': key}
        response   = requests.post(ocr_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    # get response
    analysis = response.json()
    # Extract the word bounding boxes and text
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info)
    s = []
    for boundingBox in word_infos:
        s.append(boundingBox['text'].lower())
    return s

# text recognition: similar to OCR but trained with updated recognition models and executes asynchronously
# https://goo.gl/A8jsz1
# https://goo.gl/xxfwRA
def recognize_handwritten_text(vision_base_url, key, image_url, local_image=False):
    """ return a list of lowercase strings, may contain special characters """
    """ can take local image path or remote url """
    text_recognition_url = vision_base_url + "recognizeText"
    params  = {'mode': 'Handwritten'}
    if local_image:
        image_path = image_url
        image_data = open(image_path, 'rb').read()
        headers = {'Ocp-Apim-Subscription-Key': key,
                   'Content-Type': 'application/octet-stream'}
        response = requests.post(text_recognition_url, headers=headers, params=params, data=image_data)
    else:
        data = {'url': image_url}
        headers = {'Ocp-Apim-Subscription-Key': key}
        response = requests.post(text_recognition_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]
    # Two API calls: one to submit image for processing, another to retrieve texts found
    analysis = {}
    poll = True      # texts are not immediately available
    while (poll):
        response_final = requests.get(response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        time.sleep(1)
        if ("recognitionResult" in analysis):
            poll = False 
        if ("status" in analysis and analysis['status'] == 'Failed'):
            poll = False
    polygons = []
    if ("recognitionResult" in analysis):
        polygons = [(line["boundingBox"], line["text"])
        for line in analysis["recognitionResult"]["lines"]]
    s = []
    for result in polygons:
        s.append(result[1])
    return s


def write_to_file(vision_base_url, key, df, output_file_path, dataset): 
    if (not (isinstance(df,pd.DataFrame) and dataset in ['vizwiz','vqa'])):
        print("Check argv")
        raise ValueError
    file = open(output_file_path, 'w+')
    file.write("qid;question;ocr_text;handwritten_text\n")
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
        ocr_text   = ocr_request(vision_base_url, key, image_url, local_image)
        handwritten_text = recognize_handwritten_text(vision_base_url, key, image_url, local_image)
        result_str = "{};{};{};{}\n".format(qid,qsn,ocr_text,handwritten_text)
        file.write(result_str)
        n += 1
    file.close()
    print("OCR and handwritten text recognition results for {} written to {}".format(dataset, output_file_path))