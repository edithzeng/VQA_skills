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
import logging
from nltk.stem import PorterStemmer, WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords

# configure GPU
config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 56})
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

kfold=KFold(n_splits=10)

VOCAB_SIZE = 50000
EMBEDDING_DIM = 300

class TextFeatures():

	def __init__(self):
		pass

	def import_text_features(self):
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
		self.vizwiz_targets_train = pd.read_csv('../vizwiz_skill_typ_train.csv', dtype={'QID':str},
										delimiter=',', quotechar='"',
										engine='python', error_bad_lines=False, warn_bad_lines=False)
		self.vizwiz_targets_val = pd.read_csv('../vizwiz_skill_typ_val.csv', dtype={'QID':str},
									delimiter=',', quotechar='"', engine='python', error_bad_lines=False, warn_bad_lines=False)
		self.vqa_targets_train = pd.read_csv('../vqa_skill_typ_train.csv', dtype={'QID':str},
										engine='python', quotechar='"', error_bad_lines=False, warn_bad_lines=False)
		self.vqa_targets_val = pd.read_csv('../vqa_skill_typ_val.csv', dtype={'QID':str},
										engine='python', quotechar='"', error_bad_lines=False, warn_bad_lines=False)
	def join_feature_target(self, feature_df_text, feature_df_color, target_df):
		feature_text = copy.deepcopy(feature_df_text)
		feature_color = copy.deepcopy(feature_df_color)
		target = copy.deepcopy(target_df)
		# text features 
		feature_text.rename({'qid': 'QID'}, axis=1, inplace=True)
		feature_text.set_index('QID', inplace=True)
		# color features
		feature_color.rename({'qid': 'QID'}, axis=1, inplace=True)
		feature_color.set_index('QID', inplace=True)
		# join features
		features = feature_text.join(feature_color[['descriptions','tags','dominant_colors']],
								   on='QID',
								   how='outer')
		# join features with target
		target = target[['QID', 'IMG', 'QSN', 'TXT', 'OBJ', 'COL', 'CNT', 'OTH']]
		target.set_index('QID', inplace=True)
		target = target.astype(dtype=str)
		df = target.join(features, on='QID', how='inner')
		df['descriptions'].astype(list)
		print("Joined features with skill labels.")
		return df
	def create_df(self):
		self.vizwiz_train = self.join_feature_target(self.vizwiz_features_train_text, self.vizwiz_features_train_color, 
			self.vizwiz_targets_train)
		self.vizwiz_val   = self.join_feature_target(self.vizwiz_features_val_text, self.vizwiz_features_val_color,
			self.vizwiz_targets_val)
		self.vqa_train    = self.join_feature_target(self.vqa_features_train_text, self.vqa_features_train_color, 
			self.vqa_targets_train)
		self.vqa_val      = self.join_feature_target(self.vqa_features_val_text, self.vqa_features_val_color,
			self.vqa_targets_val)
		print("VizWiz training shape:", self.vizwiz_train.shape)
		print("VQA training shape:", self.vqa_train.shape)
		print("Total training rows:", self.vizwiz_train.shape[0] + self.vqa_train.shape[0])
		print("VizWiz validation shape:{}\nVQA validation shape:{}".format(self.vizwiz_val.shape,self.vqa_val.shape))
		print("Total validation rows:", self.vizwiz_val.shape[0] + self.vqa_val.shape[0])

	def choose_dataset(self, dataset):
		if dataset == 'vizwiz':
			self.train = self.vizwiz_train
			self.val   = self.vizwiz_val
		elif dataset == 'vqa':
			self.train = self.vqa_train
			self.val   = self.vqa_val
		elif dataset == 'both':
			self.train = pd.concat([vizwiz_train, vqa_train], axis=0)
			self.val   = pd.concat([vizwiz_val, vqa_val], axis=0)
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
		if self.train is not None and self.val is not None:
			self.txt_train = self.train['TXT'].values
			self.col_train = self.train['COL'].values
			self.cnt_train = self.train['CNT'].values
			self.txt_val   = self.val['TXT'].values.astype('float32')
			self.col_val   = self.val['COL'].values.astype('float32')
			self.cnt_val   = self.val['CNT'].values.astype('float32')
	def set_features(self, feature_columns):
		features_train = self.preprocess_text(self.train[feature_columns])
		features_val   = self.preprocess_text(self.val[feature_columns])
		self.features_train = self.remove_stop_words(features_train)
		self.features_val   = self.remove_stop_words(features_val)
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