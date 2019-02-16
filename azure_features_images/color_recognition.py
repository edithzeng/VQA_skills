from util_API_color_recognition import *

# connection setup - color in image
key = str(input("Enter computer vision image analysis key:"))
vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

# load color recognition entries in training sets
vizwiz_train = pd.read_csv("../../vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vqa_train = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vizwiz_train_color = vizwiz_train.loc[vizwiz_train['COL'] == 1]
vqa_train_color = vqa_train.loc[vqa_train['COL'] == 1]

# load color recognition entries in validation sets 
vizwiz_val = pd.read_csv("../../vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')
vqa_val = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
vizwiz_val_color = vizwiz_val.loc[vizwiz_val['COL'] == 1]
vqa_val_color = vqa_val.loc[vqa_val['COL'] == 1]


# label training sets
write_to_file(vision_base_url, key, vqa_train, "vqa_train_color_recognition.csv", 'vqa')
write_to_file(vision_base_url, key, vizwiz_train, "vizwiz_train_color_recognition.csv", 'vizwiz')

# label validation sets
write_to_file(vision_base_url, key, vqa_val, "vqa_val_color_recognition.csv", 'vqa')
write_to_file(vision_base_url, key, vizwiz_train, "vizwiz_val_color_recognition.csv", 'vizwiz')