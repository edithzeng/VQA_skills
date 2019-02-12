"""
Call MS Azure text analytics API to get key phrases
Then writes to a local .csv
"""

import sys
from text_api import *

# connection setup - text analytics
key = str(input("Enter text analytics key:"))
url = 'https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/keyPhrases'

# load data
vqa_train = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vqa_val = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
vqa_test = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_test.csv', skipinitialspace=True, engine='python')
vizwiz_train = pd.read_csv('../../vizwiz_skill_typ_train.csv', skipinitialspace=True, engine='python')
vizwiz_val = pd.read_csv('../../vizwiz_skill_typ_val.csv', skipinitialspace=True, engine='python')
vizwiz_test = pd.read_csv('../../vizwiz_skill_typ_test.csv', skipinitialspace=True, engine='python')

# get key phrases and write to csv files
get_azure_keyphrases(vqa_train, "vqa_train_keyphrases.csv", key, url)
get_azure_keyphrases(vqa_val, "vqa_val_keyphrases.csv", key, url)
get_azure_keyphrases(vizwiz_train, "vizwiz_train_keyphrases.csv", key, url)
get_azure_keyphrases(vizwiz_val, "vizwiz_val_keyphrases.csv", key, url)