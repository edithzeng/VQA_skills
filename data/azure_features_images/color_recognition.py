import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util_API_color_recognition import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import args

# connection setup - color in image
key = str(input("Computer vision image analysis key for microsoft VM:"))
vision_base_url = "https://eastus.api.cognitive.microsoft.com/vision/v2.0/"

# training sets
vizwiz_train = pd.read_csv(args.vizwiz_qsn_train, skipinitialspace=True, engine='python')
vqa_train = pd.read_csv(args.vqa_qsn_train, skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_train, "vqa_train_color_recognition.csv", 'vqa')
write_to_file(vision_base_url, key, vizwiz_train, "vizwiz_train_color_recognition.csv", 'vizwiz')

# validation sets
vizwiz_val = pd.read_csv(args.vizwiz_qsn_val, skipinitialspace=True, engine='python')
vqa_val = pd.read_csv(args.vqa_qsn_val, skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_val, "vqa_val_color_recognition.csv", 'vqa')
write_to_file(vision_base_url, key, vizwiz_val, "vizwiz_val_color_recognition.csv", 'vizwiz')

# test 
vizwiz_test = pd.read_csv(args.vizwiz_qsn_test, skipinitialspace=True, engine='python')
vqa_test = pd.read_csv(args.vqa_qsn_test, skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_test, "vqa_test_color_recognition.csv", 'vqa')
write_to_file(vision_base_url, key, vizwiz_test, "vizwiz_test_color_recognition.csv", 'vizwiz')
