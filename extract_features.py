import pandas as pd 
import random
import numpy as np
import copy
import time
import re
import os

from contractions import *

import h5py
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import gzip
import pickle
import requests
import gzip

import keras
import tensorflow as tf
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
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import logging
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')

# configure GPU
config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 56})
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

kfold=KFold(n_splits=10)

# VOCAB_SIZE = 50000
VOCAB_SIZE = 3000
#EMBEDDING_DIM = 300
EMBEDDING_DIM = 100




class Features():

    def __init__(self, dataset, qsn_x_path='azure_features_images/two_vote_threshold',
                img_x_path='nn_features_images', target_path='../data/two_vote_threshold'):
        # training on datasets separately
        if dataset not in ['vizwiz', 'vqa']:
            raise ValueError("Specify 'vizwiz' or 'vqa'")
        self.dataset = dataset

        # path to files with question-based features extracted with Azure
        self.__check_path(qsn_x_path)
        self.qsn_x_path = qsn_x_path
        # self.check_path(img_x_path)
        self.img_x_path = img_x_path
        self.__check_path(target_path)
        self.target_path = target_path
  
    def __check_path(self, _dir):
        assert (os.path.isdir(_dir) and len(os.listdir(_dir)) > 0)

    def import_vizwiz_question(self):
        self.vizwiz_features_train_color = pd.read_csv(os.path.join(self.qsn_x_path, 'vizwiz_train_color_recognition.csv'),
                                    delimiter=';', engine='python', 
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'tags':list, 'dominant_colors':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_train_text = pd.read_csv(os.path.join(self.qsn_x_path, 'vizwiz_train_text_recognition.csv'),
                                    delimiter=';', engine='python', 
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'ocr_text':list, 'handwritten_text':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_val_color = pd.read_csv(os.path.join(self.qsn_x_path, 'vizwiz_val_color_recognition.csv'),
                                    delimiter=';', engine='python',
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'tags':list, 'dominant_colors':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_features_val_text = pd.read_csv(os.path.join(self.qsn_x_path, 'vizwiz_val_text_recognition.csv'),
                                    delimiter=';', engine='python', 
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'ocr_text':list, 'handwritten_text':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_targets_train = pd.read_csv(os.path.join(self.target_path, 'vizwiz_skill_typ_train.csv'), dtype={'QID':str},
                                    delimiter=',', quotechar='"',
                                    engine='python', error_bad_lines=False, warn_bad_lines=False)
        self.vizwiz_targets_val = pd.read_csv(os.path.join(self.target_path, 'vizwiz_skill_typ_val.csv'), dtype={'QID':str},
                                    delimiter=',', quotechar='"', engine='python', error_bad_lines=False, warn_bad_lines=False)
    
    def import_vizwiz_image(self):  # image-based features extracted with resnet
        self.vizwiz_features_train_image = h5py.File("./nn_features_images/vizwiz_image_feature_train.hdf5", 'r')
        self.vizwiz_features_val_image   = h5py.File("./nn_features_images/vizwiz_image_feature_val.hdf5", 'r')

    def import_vqa_question(self):   # question-based features
        self.vqa_features_train_color = pd.read_csv(os.path.join(self.qsn_x_path, 'vqa_train_color_recognition.csv'),
                                    delimiter=';', engine='python', 
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'tags':list, 'dominant_colors':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_train_text = pd.read_csv(os.path.join(self.qsn_x_path, 'vqa_train_text_recognition.csv'),
                                    delimiter=';', engine='python', 
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'ocr_text':list, 'handwritten_text':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_val_color = pd.read_csv(os.path.join(self.qsn_x_path, 'vqa_val_color_recognition.csv'),
                                    delimiter=';', engine='python',
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'tags':list, 'dominant_colors':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_features_val_text = pd.read_csv(os.path.join(self.qsn_x_path, 'vqa_val_text_recognition.csv'),
                                    delimiter=';', engine='python', 
                                    dtype={'qid':str, 'question':str, 'descriptions':list, 'ocr_text':list, 'handwritten_text':list},
                                    quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_targets_train = pd.read_csv(os.path.join(self.target_path, 'vqa_skill_typ_train.csv'), dtype={'QID':str},
                                    engine='python', quotechar='"', error_bad_lines=False, warn_bad_lines=False)
        self.vqa_targets_val   = pd.read_csv(os.path.join(self.target_path, 'vqa_skill_typ_val.csv'), dtype={'QID':str},
                                    engine='python', quotechar='"', error_bad_lines=False, warn_bad_lines=False)
    def import_vqa_image(self):
        self.vqa_features_train_image    = h5py.File("./nn_features_images/vqa_image_feature_train.hdf5", 'r')
        self.vqa_feature_val_image       = h5py.File("./nn_features_images/vqa_image_feature_val.hdf5", 'r')

    def import_features(self, image_features=False):
        if self.dataset == 'vizwiz':
            self.import_vizwiz_question()
            if image_features:
                self.import_vizwiz_image()
        elif self.dataset == 'vqa':
            self.import_vqa_question()
            if image_features:
                self.import_vqa_image()

    def join_question_feature_target(self, feature_df_text, feature_df_color, target_df):
        feature_text = copy.deepcopy(feature_df_text)
        feature_color = copy.deepcopy(feature_df_color)
        target = copy.deepcopy(target_df)
        # text recognition features 
        feature_text.rename({'qid': 'QID'}, axis=1, inplace=True)
        feature_text.set_index('QID', inplace=True)
        # color recognition features
        feature_color.rename({'qid': 'QID'}, axis=1, inplace=True)
        feature_color.set_index('QID', inplace=True)
        # join question features
        features = feature_text.join(feature_color[['descriptions','tags','dominant_colors']],
                                   on='QID', how='outer')
        # join question features with skill labels
        target = target[['QID', 'IMG', 'QSN', 'TXT', 'OBJ', 'COL', 'CNT', 'OTH']]
        target.set_index('QID', inplace=True)
        target = target.astype(dtype=str)
        df = target.join(features, on='QID', how='inner')
        df['descriptions'].astype('category')
        return df


    def concat_image_features(self):
        if self.dataset == 'vizwiz':
            train = self.vizwiz_features_train_image
            val = self.vizwiz_features_val_image
        else:
            train = self.vqa_features_train_image
            val = self.vqa_features_val_image
        df_train = copy.deepcopy(question_features_train_df)
        df_train['image_features'] = ""
        df_val = copy.deepcopy(question_features_val_df)
        df_val['image_features'] = ""
        for i in range(len(df_train)):
            row = df_train.iloc[i,:]
            qid = str(row['QID'])
            df_train.iloc[i, len(df_train.columns)-1] = train[qid]
        for i in range(len(df_val)):
            row = df_val.iloc[i,:]
            qid = str(row['QID'])
            df_val.iloc[i, len(df_val.columns)-1] = val[qid]
        self.image_features_df_train = df_train
        self.image_features_df_val = df_val
        return df_train['image_features'].values, df_val['image_features'].values


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


    def set_features(self, feature_columns):
        X_train = self.preprocess_text(self.question_features_train_df[feature_columns])
        X_val   = self.preprocess_text(self.question_features_val_df[feature_columns])
        self.question_features_train = self.remove_stop_words(X_train)
        self.question_features_val   = self.remove_stop_words(X_val)


    def create_question_feature_df(self):
        if self.dataset == 'vizwiz':
            self.question_features_train_df = self.join_question_feature_target(self.vizwiz_features_train_text, self.vizwiz_features_train_color, 
                self.vizwiz_targets_train)
            self.question_features_val_df = self.join_question_feature_target(self.vizwiz_features_val_text, self.vizwiz_features_val_color, 
                self.vizwiz_targets_val)
        elif self.dataset == 'vqa':
            self.question_features_train_df = self.join_question_feature_target(self.vqa_features_train_text, self.vqa_features_train_color, 
                self.vqa_targets_train)
            self.question_features_val_df = self.join_question_feature_target(self.vqa_features_val_text, self.vqa_features_val_color,
                self.vqa_targets_val)


    def set_targets(self):
        if self.question_features_train_df is not None and self.question_features_val_df is not None:
            self.txt_train = self.question_features_train_df['TXT'].values
            self.col_train = self.question_features_train_df['COL'].values
            self.cnt_train = self.question_features_train_df['CNT'].values
            self.txt_val   = self.question_features_val_df['TXT'].values.astype('float32')
            self.col_val   = self.question_features_val_df['COL'].values.astype('float32')
            self.cnt_val   = self.question_features_val_df['CNT'].values.astype('float32')


    def get_word_embedding(self, pretrained_embedding, 
        googlenews_corpus='/anaconda/envs/py35/lib/python3.5/site-packages/gensim/test/test_data/GoogleNews-vectors-negative300.bin'):
        # tokenize text features
        tok = Tokenizer(num_words=VOCAB_SIZE, 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                        lower=True,
                        split=" ")
        tok.fit_on_texts(self.question_features_train)
        # create training and validation sequences
        MAX_DOC_LEN = 40
        train_seq   = tok.texts_to_sequences(self.question_features_train)
        val_seq     = tok.texts_to_sequences(self.question_features_val)
        # pad training and validation sequences
        train_seq        = sequence.pad_sequences(train_seq, maxlen=MAX_DOC_LEN)
        val_seq          = sequence.pad_sequences(val_seq, maxlen=MAX_DOC_LEN)
        # standardize training and testing features
        sc = StandardScaler()
        train_seq = sc.fit_transform(train_seq)
        val_seq = sc.transform(val_seq)
        self.train_seq = train_seq
        self.val_seq = val_seq
        # punkt sentence level tokenizer
        sent_lst = [] 
        for doc in self.question_features_train:
            sentences = nltk.tokenize.sent_tokenize(doc)
            for sent in sentences:
                word_lst = [w for w in nltk.tokenize.word_tokenize(sent) if w.isalnum()]
                sent_lst.append(word_lst)
        if pretrained_embedding:
            corpus = googlenews_corpus
            # load pre-trained word2vec on GoogleNews (https://code.google.com/archive/p/word2vec/)
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(datapath(corpus), binary=True)
        else:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            word2vec_model = gensim.models.Word2Vec(sentences=sent_lst, min_count=6, size=EMBEDDING_DIM, sg=1, workers=os.cpu_count())
        # get word vetors
        embeddings_index = {}
        for word in word2vec_model.wv.vocab:
            coefs = np.asarray(word2vec_model.wv[word], dtype='float32')
            embeddings_index[word] = coefs
        print('Total %s word vectors' % len(embeddings_index))
        # Initial word embedding
        embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
        for word, i in tok.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None and i < VOCAB_SIZE:
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix
        return self.embedding_matrix
