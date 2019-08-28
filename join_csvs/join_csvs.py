""" join two csvs and change binary flags """
""" no need to update features """

import pandas as pd
import numpy as np
import os 

# new csv
vizwiz_train_new = pd.read_csv('../../data/three_vote_threshold/vizwiz_skill_typ_train.csv', skipinitialspace=True, engine='python')
vizwiz_val_new = pd.read_csv('../../data/three_vote_threshold/vizwiz_skill_typ_val.csv', skipinitialspace=True, engine='python')
vqa_train_new = pd.read_csv('../../data/three_vote_threshold/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vqa_val_new = pd.read_csv('../../data/three_vote_threshold/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')

# old csv
vizwiz_train_old = pd.read_csv("../../data/two_vote_threshold/vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vizwiz_val_old = pd.read_csv("../../data/two_vote_threshold/vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')
vqa_train_old = pd.read_csv('../../data/two_vote_threshold/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vqa_val_old = pd.read_csv('../../data/two_vote_threshold/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')


def _join(new_csv, old_csv, output_csv):
    
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
    
    # save to new csv
    df.to_csv(output_csv)
    print("written to", output_csv)
    
############
os.mkdir('cross_referenced_data')
_join(vqa_train_new, vqa_train_old, './cross_referenced_data/vqa_skill_typ_train.csv')
_join(vqa_val_new, vqa_val_old, './cross_referenced_data/vqa_skill_typ_val.csv')
_join(vizwiz_train_new, vizwiz_train_old, './cross_referenced_data/vizwiz_skill_typ_train.csv')
_join(vizwiz_val_new, vizwiz_val_old, './cross_referenced_data/vizwiz_skill_typ_val.csv')