import os
import sys
# paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

vizwiz_qsn_train = os.path.join(DATA_DIR, 'vizwiz_skill_typ_train.csv')
vizwiz_qsn_val = os.path.join(DATA_DIR, 'vizwiz_skill_typ_val.csv')
vizwiz_qsn_test = os.path.join(DATA_DIR, 'vizwiz_skill_typ_test.csv')

vqa_qsn_train = os.path.join(DATA_DIR, 'vqa_skill_typ_train.csv') 
vqa_qsn_val = os.path.join(DATA_DIR, 'vqa_skill_typ_val.csv')
vqa_qsn_test = os.path.join(DATA_DIR, 'vqa_skill_typ_test.csv')

vizwiz_sets = [vizwiz_qsn_train, vizwiz_qsn_val, vizwiz_qsn_test]
vqa_sets = [vqa_qsn_train, vqa_qsn_val, vqa_qsn_test]
