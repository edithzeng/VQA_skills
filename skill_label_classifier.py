import pandas as pd 
import random
import numpy as np
import copy
import time
import re
import os

from contractions import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import gzip
import pickle
import requests
import gzip

os.environ['KERAS_BACKEND']='cntk'
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # disable GPU
os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential, load_model
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.initializers import he_normal
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, Activation
from keras.callbacks import History, CSVLogger, EarlyStopping, LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.utils import to_categorical

import nltk
import gensim
import logging
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
# config = tf.ConfigProto(device_count={'GPU': -1, 'CPU': 1})
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

kfold=KFold(n_splits=10)

VOCAB_SIZE = 3000
EMBEDDING_DIM = 300

class SkillClassifier():

    """ A wrapper for a simple binary classifier for each skill label. """

    def __init__(self, name=None):
        self.name = name 

    def import_data(self):
        self.vizwiz_features_train_color = pd.read_csv('azure_features_images/data/vizwiz_train_color_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'tags':list, 'dominant_colors':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_train_text = pd.read_csv('azure_features_images/data/vizwiz_train_text_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'ocr_text':list, 'handwritten_text':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_val_color = pd.read_csv('azure_features_images/data/vizwiz_val_color_recognition.csv',
                                    delimiter=';', engine='python',
                                    dtype={'qid':str, 'question':str, 'descriptions':list,
                                        'tags':list, 'dominant_colors':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_val_text = pd.read_csv('azure_features_images/data/vizwiz_val_text_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'ocr_text':list, 'handwritten_text':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_test_color = pd.read_csv('azure_features_images/data/vizwiz_test_color_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'ocr_text':list, 'handwritten_text':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_test_text = pd.read_csv('azure_features_images/data/vizwiz_test_text_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'ocr_text':list, 'handwritten_text':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_train_color = pd.read_csv('azure_features_images/data/vqa_train_color_recognition.csv',
                                    delimiter=';', engine='python', 
                                    dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'tags':list, 'dominant_colors':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_train_text = pd.read_csv('azure_features_images/data/vqa_train_text_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'ocr_text':list, 'handwritten_text':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_val_color = pd.read_csv('azure_features_images/data/vqa_val_color_recognition.csv',
                                    delimiter=';', engine='python',
                                        dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'tags':list, 'dominant_colors':list},
                                        quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_val_text = pd.read_csv('azure_features_images/data/vqa_val_text_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'ocr_text':list, 'handwritten_text':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_test_color = pd.read_csv('azure_features_images/data/vqa_test_color_recognition.csv',
                                    delimiter=';', engine='python',
                                        dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'tags':list, 'dominant_colors':list},
                                        quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_test_text = pd.read_csv('azure_features_images/data/vqa_test_text_recognition.csv',
                                      delimiter=';', engine='python', 
                                      dtype={'qid':str, 'question':str, 'descriptions':list,
                                            'ocr_text':list, 'handwritten_text':list},
                                      quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_targets_train = pd.read_csv('../data/three_vote_threshold/vizwiz_skill_typ_train.csv', dtype={'QID':str},
                                        delimiter=',', quotechar='"',
                                        engine='python', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_targets_val = pd.read_csv('../data/three_vote_threshold/vizwiz_skill_typ_val.csv', dtype={'QID':str},
                            delimiter=',', quotechar='"', engine='python', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_targets_test = pd.read_csv('../data/three_vote_threshold/vizwiz_skill_typ_test.csv', dtype={'QID':str},
                            delimiter=',', quotechar='"', engine='python', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_targets_train = pd.read_csv('../data/three_vote_threshold/vqa_skill_typ_train.csv', dtype={'QID':str},
                            engine='python', quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_targets_val = pd.read_csv('../data/three_vote_threshold/vqa_skill_typ_val.csv', dtype={'QID':str},
                            engine='python', quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_targets_test = pd.read_csv('../data/three_vote_threshold/vqa_skill_typ_test.csv', dtype={'QID':str},
                            engine='python', quotechar='"', error_bad_lines=False, warn_bad_lines=False)


    def join_feature_target(self, feature_df_text, feature_df_color, target_df):
        feature_text = copy.deepcopy(feature_df_text)
        feature_color = copy.deepcopy(feature_df_color)
        target = copy.deepcopy(target_df)
        # text
        feature_text.rename({'qid': 'QID'}, axis=1, inplace=True)
        feature_text.set_index('QID', inplace=True)
        # color
        feature_color.rename({'qid': 'QID'}, axis=1, inplace=True)
        feature_color.set_index('QID', inplace=True)
        # join
        features = feature_text.join(feature_color[['descriptions','tags','dominant_colors']],
                                   on='QID',
                                   how='outer')
        # add label skill categories
        target = target[['QID', 'IMG', 'QSN', 'TXT', 'OBJ', 'COL', 'CNT', 'OTH']]
        target.set_index('QID', inplace=True)
        target = target.astype(dtype=str)
        df = target.join(features, on='QID', how='inner')
        df['descriptions'].astype(np.ndarray)
        print("Joined features with skill labels.")
        return df

    def __create_binary_flags(self):
        # based on 3 vote threshold
        create_flag = lambda x: 1. if int(float(x)) >= 3 else 0.
        dsets = [self.vizwiz_train, self.vizwiz_val, self.vizwiz_test,
                self.vqa_train, self.vqa_val, self.vqa_test]
        for d in dsets:
            d["COL_FLAG"] = d["COL"].apply(create_flag)
            d["TXT_FLAG"] = d["TXT"].apply(create_flag)
            d["CNT_FLAG"] = d["CNT"].apply(create_flag)

    def create_df(self):
        """ concatenate features & binary flags """
        self.vizwiz_train = self.join_feature_target(self.vizwiz_features_train_text, self.vizwiz_features_train_color, 
            self.vizwiz_targets_train)
        self.vizwiz_val   = self.join_feature_target(self.vizwiz_features_val_text, self.vizwiz_features_val_color,
            self.vizwiz_targets_val)
        self.vizwiz_test  = self.join_feature_target(self.vizwiz_features_test_text, self.vizwiz_features_test_color,
            self.vizwiz_targets_test)
        self.vqa_train    = self.join_feature_target(self.vqa_features_train_text, self.vqa_features_train_color, 
            self.vqa_targets_train)
        self.vqa_val      = self.join_feature_target(self.vqa_features_val_text, self.vqa_features_val_color,
            self.vqa_targets_val)
        self.vqa_test     = self.join_feature_target(self.vqa_features_test_text, self.vqa_features_test_color,
            self.vqa_targets_test)
        
        self.__create_binary_flags()
        
        print("VizWiz training shape:", self.vizwiz_train.shape)
        print("VQA training shape:", self.vqa_train.shape)
        print("Total training rows:", self.vizwiz_train.shape[0] + self.vqa_train.shape[0])
        print("VizWiz validation shape:{}\nVQA validation shape:{}".format(self.vizwiz_val. shape,self.vqa_val.shape))
        print("Total validation rows:", self.vizwiz_val.shape[0] + self.vqa_val.shape[0])
        print("VizWiz test shape:{}\nVQA test shape:{}".format(self.vizwiz_test.shape, self.vqa_test.shape))
        print("Total test rows:", self.vizwiz_test.shape[0] + self.vqa_test.shape[0])
                
    def choose_dataset(self, dataset):
        if dataset == 'vizwiz':
            self.train = self.vizwiz_train
            self.val   = self.vizwiz_val
            self.test  = self.vizwiz_test
        elif dataset == 'vqa':
            self.train = self.vqa_train
            self.val   = self.vqa_val
            self.test  = self.vqa_test
        elif dataset == 'both':
            self.train = pd.concat([vizwiz_train, vqa_train], axis=0)
            self.val   = pd.concat([vizwiz_val, vqa_val], axis=0)
            self.test  = pd.concat([vizwiz_test, vqa_test], axis=0)
        else:
            raise ValueError("Specify dataset: 'vizwiz', 'vqa' or 'both'")
        print("Training: {}\nValidation: {}".format(self.train.shape, self.val.shape))


    def preprocess_text(self, feature_df_subset):
        """ output an nparray with single document per data point """
        ip = copy.deepcopy(feature_df_subset).values
        op = []
        for i in range(ip.shape[0]):
            doc       =  ""
            for j in range(ip.shape[1]):                
                s     =  str(ip[i][j]).lower()
                # expand contractions
                for cont in contractions:
                    if cont in s:
                        s = s.replace(cont, contractions[cont])
                # remove special chars
                s     =  s.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+'"}) + " "
                # remove tags in descriptions
                if feature_df_subset.columns[j] == 'descriptions':
                    s = re.sub(r'confidence\s+\d+', '', s)
                    s = re.sub(r'text', '', s)
                # Lemmatize
                s    = self.lem(s)
                doc  += s
            op.append(doc)
        op = np.asarray(op)
        return op


    def set_targets(self):
        """ the goal is to predict a binary flag, not the number of votes per category """
        if self.train is not None and self.val is not None:
            self.txt_train = self.train['TXT_FLAG'].values.astype('float32')
            self.col_train = self.train['COL_FLAG'].values.astype('float32')
            self.cnt_train = self.train['CNT_FLAG'].values.astype('float32')

            self.txt_val   = self.val['TXT_FLAG'].values.astype('float32')
            self.col_val   = self.val['COL_FLAG'].values.astype('float32')
            self.cnt_val   = self.val['CNT_FLAG'].values.astype('float32')
            
            self.txt_test  = self.test['TXT_FLAG'].values.astype('float32')
            self.col_test  = self.test['COL_FLAG'].values.astype('float32')
            self.cnt_test  = self.test['CNT_FLAG'].values.astype('float32')


    def set_features(self, feature_columns):
        features_train = self.preprocess_text(self.train[feature_columns])
        features_val   = self.preprocess_text(self.val[feature_columns])
        features_test  = self.preprocess_text(self.test[feature_columns])
        self.features_train = self.remove_stop_words(features_train)
        self.features_val   = self.remove_stop_words(features_val)
        self.features_test  = self.remove_stop_words(features_test)


    def lem(self, s):
        arr = s.split(" ")
        lem = WordNetLemmatizer()
        op = ""
        for w in arr:
            word = lem.lemmatize(w) + ' '
            op += word
        return op

    def remove_stop_words(self, features):
        stop_words = set(stopwords.words('english'))
        cleansed = [w for w in features if not w in stop_words]
        return cleansed


###############################################################################
def lstm_create_train(MAX_DOC_LEN, train_seq, embedding_matrix,
    train_labels, val_data, learning_rate, lstm_dim, batch_size, 
    num_epochs, optimizer_param, regularization=1e-7, n_classes=3, verbose=0):

    l2_reg = regularizers.l2(regularization)

    # init model
    embedding_layer = Embedding(VOCAB_SIZE+1, EMBEDDING_DIM,
                                input_length=MAX_DOC_LEN,
                                trainable=False,
                                mask_zero=False,
                                embeddings_regularizer=l2_reg,
                                weights=[embedding_matrix])
    model = Sequential()
    model.add(embedding_layer)
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(activation='tanh', units=lstm_dim, return_sequences=True)))
    model.add(Bidirectional(LSTM(activation='tanh', units=lstm_dim, dropout=0.5, return_sequences=True)))
    model.add(Bidirectional(LSTM(activation='tanh', units=lstm_dim)))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer_param,
                  metrics=['acc'])

    history = History()
    logfile = './LSTM/{}_{}_{}_{}.log'.format(learning_rate, regularization, batch_size, num_epochs)
    csv_logger = CSVLogger(logfile, separator=',', append=True)
    # checkpoint = ModelCheckpoint(filepath='./LSTM/weights.hdf5', verbose=1, save_best_only=True)
    # exponential scheduling (Andrew Senior et al., 2013) for Nesterov
    scheduler = LearningRateScheduler(lambda x: learning_rate*10**(-1*x/64), verbose=0)
    # stop = EarlyStopping(patience=200)
    print("Log file:", logfile)

    t1 = time.time()
    model.fit(train_seq,
              train_labels.astype('float32'),
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=val_data,
              shuffle=True,
              callbacks=[scheduler, history, csv_logger],
              verbose=verbose)
    t2 = time.time()
    # save hdf5
    model.save('./LSTM/{}_{}_{}_{}_model.h5'.format(learning_rate, regularization, batch_size, num_epochs))
    #np.savetxt('./LSTM/{}_{}_{}_{}_time.txt'.format(learning_rate, regularization, batch_size, num_epochs), 
    #           [regularization, (t2-t1) / 3600])
    with open('./LSTM/{}_{}_{}_{}_history.txt'.format(learning_rate, regularization, batch_size, num_epochs), "w") as res_file:
        res_file.write(str(history.history))
    return model, history
