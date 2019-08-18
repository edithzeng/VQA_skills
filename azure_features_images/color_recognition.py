from util_API_color_recognition import *

# connection setup - color in image
key = str(input("Enter computer vision image analysis key for microsoft VM:"))
vision_base_url = "https://eastus.api.cognitive.microsoft.com/vision/v2.0/"

# training sets - done
#vizwiz_train = pd.read_csv("../../data/three_vote_threshold/vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
#vqa_train = pd.read_csv('../../data/three_vote_threshold/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
#write_to_file(vision_base_url, key, vqa_train, "vqa_train_color_recognition.csv", 'vqa')
#write_to_file(vision_base_url, key, vizwiz_train, "vizwiz_train_color_recognition.csv", 'vizwiz')

# validation sets
vizwiz_val = pd.read_csv("../../data/three_vote_threshold/vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')
vqa_val = pd.read_csv('../../data/three_vote_threshold/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_val, "vqa_val_color_recognition.csv", 'vqa')
write_to_file(vision_base_url, key, vizwiz_val, "vizwiz_val_color_recognition.csv", 'vizwiz')

# test 
vizwiz_test = pd.read_csv("../../data/three_vote_threshold/izwiz_skill_typ_test.csv", skipinitialspace=True, engine='python')
vqa_test = pd.read_csv('../../data/three_vote_threshold/vqa_skill_typ_test.csv', skipinitialspace=True, engine='python')
write_to_file(vision_base_url, key, vqa_test, "vqa_test_color_recognition.csv", 'vqa')
write_to_file(vision_base_url, key, vizwiz_test, "vizwiz_test_color_recognition.csv", 'vizwiz')
