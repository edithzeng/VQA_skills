#!/usr/bin/python3
import pandas as pd
import os
import sys 
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import args

for s in args.vizwiz_sets:

    if not os.path.exists(s):
        raise FileNotFoundError(s)

    df = pd.read_csv(s, skipinitialspace=True, engine='python')

    split = df['split'].tolist()[0]
    dest_dir = os.path.abspath(f'images/vizwiz_{split}')
    os.mkdir(dest_dir)

    imgs = df['IMG'].tolist()
    base = 'https://ivc.ischool.utexas.edu/VizWiz/data/Images/'
    
    for i, img in enumerate(imgs):
        url = base + img.strip()
        r = urllib.request.urlretrieve(url, os.path.join(dest_dir, img)) 


