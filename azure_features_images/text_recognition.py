from util_API_text_recognition import *

# connection setup - image analysis
key = str(input("Enter computer vision image analysis key:"))
vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

# load questions
vizwiz = pd.read_csv("../../vizwiz_skill_typ_train.csv", skipinitialspace=True, engine='python')
vqa = pd.read_csv('../../VQA_data/questions/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
# get rows with text recognition labels
vizwiz_txt = vizwiz.loc[vizwiz['TXT']==1]
vqa_txt = vqa.loc[vqa['TXT']==1]

# call API
#print("Calling API - VQA")
#write_to_file(vision_base_url, key, vqa_txt, 'vqa_text_recognition.csv', 'vqa')
print("Calling API - VizWiz")
write_to_file(vision_base_url, key, vizwiz_txt, 'vizwiz_text_recognition.csv', 'vizwiz')