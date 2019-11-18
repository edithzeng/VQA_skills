import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util_API_text_recognition import *

parser = argparse.ArgumentParser(description='Procedure to extract relevant features for color recognition label.')

parser.add_argument('--vizwiz_qsn_train', type=str, default='../../data/questions/vizwiz_skill_typ_train.csv')
parser.add_argument('--vizwiz_qsn_val', type=str, default='../../data/questions/vizwiz_skill_typ_val.csv')
parser.add_argument('--vizwiz_qsn_test', type=str, default='../../data/questions/vqa_skill_typ_test.csv')

parser.add_argument('--vqa_qsn_train', type=str, default='../../data/questions/vqa_skill_typ_train.csv')
parser.add_argument('--vqa_qsn_val', type=str, default='../../data/questions/vqa_skill_typ_val.csv')
parser.add_argument('--vqa_qsn_test', type=str, default='../../data/questions/vqa_skill_typ_test.csv')

args = parser.parse_args

# connection setup - image analysis
key = str(input("Enter computer vision image analysis key:"))
vision_base_url = "https://eastus.api.cognitive.microsoft.com/vision/v2.0/"

# training sets
vizwiz_train = pd.read_csv(args.vizwiz_qsn_train, skipinitialspace=True, engine='python')
vqa_train = pd.read_csv(args.vqa_qsn_train, skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_train, 'vqa_train_text_recognition.csv', 'vqa')
write_to_file(vision_base_url, key, vizwiz_train, 'vizwiz_train_text_recognition.csv', 'vizwiz')

# validation sets
vizwiz_val = pd.read_csv(args.vizwiz_qsn_val, skipinitialspace=True, engine='python')
vqa_val = pd.read_csv(args.vqa_qsn_val, skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_val, 'vqa_val_text_recognition.csv', 'vqa')
write_to_file(vision_base_url, key, vizwiz_val, 'vizwiz_val_text_recognition.csv', 'vizwiz')

# test sets 
vizwiz_test = pd.read_csv(args.vizwiz_qsn_test, skipinitialspace=True, engine='python')
vqa_test = pd.read_csv(args.vqa_qsn_test, skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_test, 'vqa_test_text_recognition.csv', 'vqa')
write_to_file(vision_base_url, key, vizwiz_test, 'vizwiz_test_text_recognition.csv', 'vizwiz')
