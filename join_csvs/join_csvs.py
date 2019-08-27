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

	counterpart = new_csv.copy()

	# add new columns for 3-vote threshold
	counterpart['COL_FLAG'] = ''
	counterpart['TXT_FLAG'] = ''
	counterpart['CNT_FLAG'] = ''
        counterpart['QID'] = counterpart['QID'].astype(str)
	counterpart['QID'] = counterpart['QID'].str.lower()

	# get indices
	col_i = counterpart.columns.get_loc('COL_FLAG')
	txt_i = counterpart.columns.get_loc('TXT_FLAG')
	cnt_i = counterpart.columns.get_loc('CNT_FLAG')

	for row_i, row in old_csv.iterrows():
		# identify the same vq in old csv
		qid = row['QID'].lower()
		record = counterpart.loc[counterpart['QID'] == qid]
		if len(record) == 0:
			print('not found')
			continue
		# identify the number of vote to create new binary flags
		old_row_i = record.index.values.astype(int)[0]
		counterpart[old_row_i, col_i] = 1 if row['COL'] >= 3 else 0
		counterpart[old_row_i, txt_i] = 1 if row['TXT'] >= 3 else 0
		counterpart[old_row_i, cnt_i] = 1 if row['CNT'] >= 3 else 0

	# save to new csv
	counterpart.to_csv(output_csv)
	print('Written to', output_csv)

############
#os.mkdir('cross_referenced_data')
# _join(vizwiz_train_new, vizwiz_train_old, './cross_referenced_data/vizwiz_skill_typ_train.csv')
_join(vizwiz_val_new, vizwiz_val_old, './cross_referenced_data/vizwiz_skill_typ_val.csv')
_join(vqa_train_new, vqa_train_old, './cross_referenced_data/vqa_skill_typ_train.csv')
_join(vqa_val_new, vqa_val_old, './cross_referenced_data/vqa_skill_typ_val.csv')
