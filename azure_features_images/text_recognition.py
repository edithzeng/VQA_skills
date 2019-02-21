from util_API_text_recognition import *

# connection setup - image analysis
key = str(input("Enter computer vision image analysis key:"))
vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

""" Labeling without knowing skills (using all labels) """
# get previously eliminated records
# training sets
vizwiz = pd.read_csv("../../vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vqa = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vizwiz_txt_other = vizwiz.loc[vizwiz['TXT'] != 1]
vqa_txt_other = vqa.loc[vqa['TXT'] != 1]

print("Processing - training - VQA others (non-text labels)")
write_to_file(vision_base_url, key, vqa_txt_other, 'vqa_train_text_recognition_other.csv', 'vqa')
print("Processing - training - VizWiz others (non-text labels)")
write_to_file(vision_base_url, key, vizwiz_txt_other, 'vizwiz_train_text_recognition_other.csv', 'vizwiz')

# validation sets
vqa_val = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
vizwiz_val = pd.read_csv("../../vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')
vqa_val_txt_other = vqa_val.loc[vqa_val['TXT'] != 1]
vizwiz_val_txt_other = vizwiz_val.loc[vizwiz_val['TXT'] != 1]

print("Processing - validation - VQA others (non-text labels)")
write_to_file(vision_base_url, key, vqa_val_txt_other, 'vqa_val_text_recognition_other.csv', 'vqa')
print("Processing - validation - VQA others (non-text labels)")
write_to_file(vision_base_url, key, vizwiz_val_txt_other, 'vizwiz_val_text_recognition_other.csv', 'vizwiz')


""" Label with known skills - complete
### training sets ###
# get rows with text recognition labels
vizwiz = pd.read_csv("../../vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vqa = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vizwiz_txt = vizwiz.loc[vizwiz['TXT']==1]
vqa_txt = vqa.loc[vqa['TXT']==1]

# call API
print("Processing - VQA")
write_to_file(vision_base_url, key, vqa_txt, 'vqa_text_recognition.csv', 'vqa')
print("Processing - VizWiz")
write_to_file(vision_base_url, key, vizwiz_txt, 'vizwiz_train_text_recognition_1.csv', 'vizwiz')

# pick up from problematic rows in vizwiz
vizwiz_2 = vizwiz[4157:]
vizwiz_txt_2 = vizwiz_2.loc[vizwiz_2['TXT'] == 1]
print("Calling API - finish text recognition for remaining images in VizWiz")
write_to_file(vision_base_url, key, vizwiz_txt_2, "vizwiz_train_recognition_2.csv", "vizwiz")

#vizwiz_3 = vizwiz[9730:]
vizwiz_txt_3 = vizwiz_3.loc[vizwiz_3['TXT'] == 1]
print("Calling API - finish text recognition for remaining images in VizWiz")
write_to_file(vision_base_url, key, vizwiz_txt_3, "vizwiz_train_text_recognition_3.csv", "vizwiz")

### validation sets ###
vqa_val = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
vizwiz_val = pd.read_csv("../../vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')

vqa_val_txt = vqa_val.loc[vqa_val['TXT'] == 1]
vizwiz_val_txt = vizwiz_val.loc[vizwiz_val['TXT'] == 1]

print("Processing text recognition - VQA validation set")
write_to_file(vision_base_url, key, vqa_val_txt, 'vqa_val_text_recognition.csv', 'vqa')

print("Processing text recognition - VizWiz validation set")
write_to_file(vision_base_url, key, vizwiz_val_txt, 'vizwiz_val_text_recognition.csv', 'vizwiz')

"""