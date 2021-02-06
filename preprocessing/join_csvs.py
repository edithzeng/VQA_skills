""" join two csvs and change binary flags 
no need to update features """
import os
import shutil
import argparser
import warnings

import pandas as pd
import numpy as np

def join(new_csv, old_csv, output_csv):
    
    df = old_csv.copy()
    new = new_csv.copy()
    
    # add columns based on 3-vote threshold
    df['QID'] = df['QID'].astype(str)
    new['QID'] = new['QID'].astype(str)
    
    # get indices
    obj = df.columns.get_loc('OBJ')
    col = df.columns.get_loc('COL')
    txt = df.columns.get_loc('TXT')
    cnt = df.columns.get_loc('CNT')
    oth = df.columns.get_loc('OTH')
    
    for i, row in df.iterrows():

        qid = row['QID']
        record = new_csv.loc[new_csv['QID'] == qid]
        if len(record) != 1:
            continue

        obj_vote = record['OBJ'].astype('int').values[0]
        col_vote = record['COL'].astype('int').values[0]
        txt_vote = record['TXT'].astype('int').values[0]
        cnt_vote = record['CNT'].astype('int').values[0]
        oth_vote = record['OTH'].astype('int').values[0]
        
        df.iloc[i, obj] = 1 if obj_vote >= 3 else 0
        df.iloc[i, col] = 1 if col_vote >= 3 else 0
        df.iloc[i, txt] = 1 if txt_vote >= 3 else 0
        df.iloc[i, cnt] = 1 if cnt_vote >= 3 else 0
        df.iloc[i, oth] = 1 if oth_vote >= 3 else 0
    
    df.to_csv(output_csv)
    print('written to', output_csv)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='join source data and save results to csv files')
    parser.add_argument(
        'new_src', 
        type=str,
        default='../../data/three_vote_threshold',
        help='path to the source directory with all csv files with 3-vote threshold'
    )
    parser.add_argument(
        'old_src', 
        type=str,
        default='../../data/two_vote_threshold',
        help='path to the source directory with all csv files with previous 2-vote threshold'
    )
    parser.add_argument(
        'dst', 
        type=str, 
        default='cross_referenced_data'
        help='path to the destination directory to write results'
    )
    args = parser.parse_args

    # new - 3 vote threshold 
    vizwiz_train_new = pd.read_csv(os.path.join(args.new_src, 'vizwiz_skill_typ_train.csv'), skipinitialspace=True, engine='python')
    vizwiz_val_new = pd.read_csv(os.path.join(args.new_src, 'vizwiz_skill_typ_val.csv'), skipinitialspace=True, engine='python')
    vqa_train_new = pd.read_csv(os.path.join(args.new_src, 'vqa_skill_typ_train.csv'), skipinitialspace=True, engine='python')
    vqa_val_new = pd.read_csv(os.path.join(args.new_src, 'vqa_skill_typ_val.csv'), skipinitialspace=True, engine='python')
    
    # old - 2 vote threshold 
    vizwiz_train_old = pd.read_csv(os.path.join(args.old_src, 'vizwiz_skill_typ_train.csv'), skipinitialspace=True, engine='python')
    vizwiz_val_old = pd.read_csv(os.path.join(args.old_src, 'vizwiz_skill_typ_val.csv'), skipinitialspace=True, engine='python')
    vqa_train_old = pd.read_csv(os.path.join(args.old_src, 'vqa_skill_typ_train.csv'), skipinitialspace=True, engine='python')
    vqa_val_old = pd.read_csv(os.path.join(args.old_src, 'vqa_skill_typ_val.csv'), skipinitialspace=True, engine='python')

    if os.path.isdir(args.dst):
        warnings.warn(f'Overwriting existing {args.dst}')  
        shutil.rmtree(args.dst)
    os.mkdir(args.dst)

    join(vqa_train_new, vqa_train_old, os.path.join(args.dst, 'vqa_skill_typ_train.csv'))
    join(vqa_val_new, vqa_val_old, os.path.join(args.dst, 'vqa_skill_typ_val.csv'))
    join(vizwiz_train_new, vizwiz_train_old, os.path.join(args.dst, 'vizwiz_skill_typ_train.csv'))
    join(vizwiz_val_new, vizwiz_val_old, os.path.join(args.dst, 'vizwiz_skill_typ_val.csv'))
