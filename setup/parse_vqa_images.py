import pandas as pd
import os

os.chdir('../..')

vqa_train = pd.read_csv('./data/three_vote_threshold/vqa_skill_typ_train.csv', skipinitialspace=True, engine='python')
vqa_val = pd.read_csv('./data/three_vote_threshold/vqa_skill_typ_val.csv', skipinitialspace=True, engine='python')
vqa_test = pd.read_csv('./data/three_vote_threshold/vqa_skill_typ_test.csv', skipinitialspace=True, engine='python')

images = []
images += list(vqa_train['IMG'].unique())
images += list(vqa_val['IMG'].unique())
images += list(vqa_test['IMG'].unique())

print(len(images))

with open('VQA_data/VQA_image_list.txt', 'w') as imglist:
    for i in images:
        imglist.write("%s\n" % i)

for root, dirs, files in os.walk('VQA_data'):
    for img in files:
        if img not in images:
            os.remove(os.path.join(root, img))
