import os
import sys

# paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

vizwiz_qsn_train = os.path.join(DATA_DIR, 'vizwiz_skill_typ_train.csv')
vizwiz_qsn_val = os.path.join(DATA_DIR, 'vizwiz_skill_typ_val.csv')
vizwiz_qsn_test = os.path.join(DATA_DIR, 'vizwiz_skill_typ_test.csv')

vqa_qsn_train = os.path.join(DATA_DIR, 'vqa_skill_typ_train.csv') 
vqa_qsn_val = os.path.join(DATA_DIR, 'vqa_skill_typ_val.csv')
vqa_qsn_test = os.path.join(DATA_DIR, 'vqa_skill_typ_test.csv')

vizwiz_sets = [vizwiz_qsn_train, vizwiz_qsn_val, vizwiz_qsn_test]
vqa_sets = [vqa_qsn_train, vqa_qsn_val, vqa_qsn_test]

# image-based features
IMG_X = os.path.join(DATA_DIR, 'azure_features_images')

vizwiz_color_recog_train = os.path.join(IMG_X, 'vizwiz_train_color_recognition.csv')
vizwiz_color_recog_val = os.path.join(IMG_X, 'vizwiz_val_color_recognition.csv')
vizwiz_color_recog_test = os.path.join(IMG_X, 'vizwiz_test_color_recognition.csv')

vizwiz_text_recog_train = os.path.join(IMG_X, 'vizwiz_train_text_recognition.csv') 
vizwiz_text_recog_val = os.path.join(IMG_X, 'vizwiz_val_text_recognition.csv')
vizwiz_text_recog_test = os.path.join(IMG_X, 'vizwiz_test_text_recognition.csv') 

vqa_color_recog_train = os.path.join(IMG_X, 'vqa_train_color_recognition.csv') 
vqa_color_recog_val = os.path.join(IMG_X, 'vqa_val_color_recognition.csv') 
vqa_color_recog_test = os.path.join(IMG_X,'vqa_test_color_recognition.csv') 
    
vqa_text_recog_train = os.path.join(IMG_X, 'vqa_train_text_recognition.csv')
vqa_text_recog_val = os.path.join(IMG_X, 'vqa_val_text_recognition.csv') 
vqa_text_recognition_test = os.path.join(IMG_X, 'vqa_test_text_recognition.csv')

# question-based features
QSN_X = os.path.join(DATA_DIR, 'azure_feature_questions')

vizwiz_keyphrase_train = os.path.join(QSN_X, 'vizwiz_train_keyphrases.csv')
vizwiz_keyphrase_val = os.path.join(QSN_X, 'vizwiz_val_keyphrases.csv')
vizwiz_keyphrase_test = os.path.join(QSN_X, 'vizwiz_test_keyphrases.csv')

vqa_keyphrase_train = os.path.join(QSN_X, 'vqa_train_keyphrases.csv')
vqa_keyphrase_val = os.path.join(QSN_X, 'vqa_val_keyphrases.csv')
vqa_keyphrase_test = os.path.join(QSN_X, 'vqa_test_keyphrases.csv')
