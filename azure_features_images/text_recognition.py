from util_API_text_recognition import *

# connection setup - image analysis
key = str(input("Enter computer vision image analysis key:"))
vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

### Label training sets ###
# get rows with text recognition labels
vizwiz = pd.read_csv("../../vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vqa = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vizwiz_txt = vizwiz.loc[vizwiz['TXT']==1]
vqa_txt = vqa.loc[vqa['TXT']==1]

# call API
#print("Calling API - VQA")
#write_to_file(vision_base_url, key, vqa_txt, 'vqa_text_recognition.csv', 'vqa')
#print("Calling API - VizWiz")
#write_to_file(vision_base_url, key, vizwiz_txt, 'vizwiz_text_recognition.csv', 'vizwiz')

# pick up from problematic rows in vizwiz
vizwiz_2 = vizwiz[4157:]
vizwiz_txt_2 = vizwiz_2.loc[vizwiz_2['TXT'] == 1]
print("Calling API - finish text recognition for remaining images in VizWiz")
write_to_file(vision_base_url, key, vizwiz_txt_2, "vizwiz_train_recognition_2.csv", "vizwiz")



### Label validation sets ###
vqa_val = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
vizwiz_val = pd.read_csv("../../vizwiz_skill_typ_val.csv", skipinitialspace=True, engine='python')

vqa_val_txt = vqa_val.loc[vqa_val['TXT'] == 1]
vizwiz_val_txt = vizwiz_val.loc[vizwiz_val['TXT'] == 1]

print("Calling API for text recognition - VQA validation set")
write_to_file(vision_base_url, key, vqa_txt_val, 'vqa_val_text_recognition.csv', 'vqa')

print("Calling API for text recognition - VizWiz validation set")
write_to_file(vision_base_url, key, vizwiz_txt_val, 'vqa_val_text_recognition.csv', 'vizwiz')