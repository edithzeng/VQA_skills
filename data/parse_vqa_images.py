#!/usr/bin/python3
import pandas as pd
import os
import sys 
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import args


for s in args.vqa_sets:

    if not os.path.exists(s):
        raise FileNotFoundError(s)

    df = pd.read_csv(s, skipinitialspace=True, engine='python')
    n = len(df)

    split = df['split'].tolist()[0]
    dest_dir = os.path.abspath(f'images/vizwiz_{split}')
    os.mkdir(dest_dir)

    imgs = df['IMG'].to_dict()
    src_dir = f'{split}2014'

    cnt = 0
    for im in os.listdir(src_dir):
        src_img = os.path.join(src_dir, imgs.pop())
        if im in imgs:            
            dest_img = os.path.join(dest_dir, im)
            shutil.move(src_img, dest_img)
            cnt += 1
        else:
            os.remove(src_img)

    assert n == cnt
