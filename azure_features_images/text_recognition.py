from util_API_text_recognition import *

# connection setup - image analysis
key = str(input("Enter computer vision image analysis key:"))
vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

# training sets
vizwiz_train = pd.read_csv("../../data/vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vqa_train = pd.read_csv('../../data/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_train, 'vqa_train_text_recognition.csv', 'vqa')
write_to_file(vision_base_url, key, vizwiz_train, 'vizwiz_train_text_recognition_other.csv', 'vizwiz')

# validation sets
vqa_val = pd.read_csv('../../data/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
vizwiz_val = pd.read_csv("../../data/vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_val, 'vqa_val_text_recognition.csv', 'vqa')
write_to_file(vision_base_url, key, vizwiz_val, 'vizwiz_val_text_recognition.csv', 'vizwiz')

# test sets 
vqa_test = pd.read_csv('../../data/vqa_skill_typ_test.csv', skipinitialspace=True, engine='python')
vizwiz_test = pd.read_csv("../../data/vizwiz_skill_typ_test.csv", skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_test, 'vqa_test_text_recognition.csv', 'vqa')
write_to_file(vision_base_url, key, vizwiz_test, 'vizwiz_test_text_recognition.csv', 'vizwiz')