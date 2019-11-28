"""
Call MS Azure text analytics API to get key phrases
Then writes to a local .csv
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from keyphrase_api_call import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import args

# connection setup - text analytics
key = str(input("Enter text analytics key:"))
url = 'https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/keyPhrases'

# load data
vizwiz_train = pd.read_csv(args.vizwiz_qsn_train, skipinitialspace=True, engine='python')
vqa_train = pd.read_csv(args.vqa_qsn_train, skipinitialspace=True, engine='python')

vizwiz_val = pd.read_csv(args.vizwiz_qsn_val, skipinitialspace=True, engine='python')
vqa_val = pd.read_csv(args.vqa_qsn_val, skipinitialspace=True, engine='python')

vizwiz_test = pd.read_csv(args.vizwiz_qsn_test, skipinitialspace=True, engine='python')
vqa_test = pd.read_csv(args.vqa_qsn_test, skipinitialspace=True, engine='python')

# get key phrases and write to csv files
get_azure_keyphrases(vqa_train, "vqa_train_keyphrases.csv", key, url)
get_azure_keyphrases(vqa_val, "vqa_val_keyphrases.csv", key, url)
get_azure_keyphrases(vqa_test, "vqa_test_keyphrases.csv", key, url)

get_azure_keyphrases(vizwiz_train, "vizwiz_train_keyphrases.csv", key, url)
get_azure_keyphrases(vizwiz_val, "vizwiz_val_keyphrases.csv", key, url)
get_azure_keyphrases(vizwiz_test, "vizwiz_test_keyphrases.csv", key, url)
