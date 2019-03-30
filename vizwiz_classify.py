from tensorflow.python.client import device_lib
import tensorflow as tf
from skill_label_classifier import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, accuracy_score

# helper function to find optimal number of PC (elbow method)
def plot_explained_variance(X_train):
    pca = PCA()
    pca_full = pca.fit(X_train)
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel("number of principal components")
    plt.ylabel("cumulative explained variance")
    plt.grid(color='grey',linestyle='-',alpha=0.2)
    plt.show()
    
def preprocess_pca(X_train, X_test, dim, r=None):
    pca = PCA(n_components=dim, random_state=r)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


experiment = SkillClassifier()
experiment.import_data()
experiment.create_df()

experiment.choose_dataset('vizwiz')
experiment.set_features(['QSN', 'descriptions', 'tags', 'dominant_colors','handwritten_text', 'ocr_text'])
experiment.set_targets()

features_train = experiment.features_train
features_val   = experiment.features_val

# check training class distribution
text_recognition_y_train = np.asarray(experiment.txt_train).astype('float32')
color_recognition_y_train = np.asarray(experiment.col_train).astype('float32')
print('Number of training samples each class: ')
print('Text recognition - 1: {} 0: {}'.format(np.count_nonzero(text_recognition_y_train), 
      len(text_recognition_y_train)-np.count_nonzero(text_recognition_y_train)))
print('Color recognition - 1:{} 0: {}'.format(np.count_nonzero(color_recognition_y_train),
     len(color_recognition_y_train)-np.count_nonzero(color_recognition_y_train)))

n_classes = 2

y_train = np.column_stack((text_recognition_y_train, color_recognition_y_train))

# check validation class distribution
text_recognition_y_val = np.asarray(experiment.txt_val).astype('float32')
color_recognition_y_val = np.asarray(experiment.col_val).astype('float32')
print('Number of validation samples each class: ')
print('Text recognition - 1: {} 0: {}'.format(np.count_nonzero(text_recognition_y_val), 
      len(text_recognition_y_val)-np.count_nonzero(text_recognition_y_val)))
print('Color recognition - 1:{} 0: {}'.format(np.count_nonzero(color_recognition_y_val),
     len(color_recognition_y_val)-np.count_nonzero(color_recognition_y_val)))

y_val = np.column_stack((text_recognition_y_val, color_recognition_y_val))

# tokenize
tok        = Tokenizer(num_words=VOCAB_SIZE, 
                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                       lower=True,
                       split=" ")
tok.fit_on_texts(features_train)

# create sequences & pad
train_seq  = tok.texts_to_sequences(features_train)
train_seq  = sequence.pad_sequences(train_seq, maxlen=MAX_DOC_LEN)
val_seq    = tok.texts_to_sequences(features_val)
val_seq    = sequence.pad_sequences(val_seq, maxlen=MAX_DOC_LEN)

# standardize training and testing features
sc = StandardScaler()
train_seq = sc.fit_transform(train_seq)
val_seq = sc.transform(val_seq)

# Set validation data tuple
val_data = (val_seq, y_val)

# punkt sentence level tokenizer
sent_lst = [] 
for doc in features_train:
    sentences = nltk.tokenize.sent_tokenize(doc)
    for sent in sentences:
        word_lst = [w for w in nltk.tokenize.word_tokenize(sent) if w.isalnum()]
        sent_lst.append(word_lst)

EMBEDDING_DIM = 300
googlenews_corpus = '/anaconda/envs/py35/lib/python3.5/site-packages/gensim/test/test_data/GoogleNews-vectors-negative300.bin'
        
# load pre-trained word2vec on GoogleNews (https://code.google.com/archive/p/word2vec/)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(datapath(googlenews_corpus),
                                                                 binary=True)
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

# PCA for dimensionality reduction
N_COMPONENTS = 40
train_seq, val_seq = preprocess_pca(train_seq, val_seq, dim=N_COMPONENTS)

# train RNN
L = 1e-1
R = 0
B = 64
E = 1000
model, history = lstm_create_train(train_seq, embedding_matrix,
                 train_labels=y_train,
                 val_data=val_data,
                 learning_rate=L,
                 lstm_dim=100,
                 batch_size=B,
                 num_epochs=E,
                 optimizer_param=SGD(lr=L, nesterov=True),
                 regularization=R, n_classes=2)
preds = model.predict(val_seq, verbose=0)
print("accuracy:", accuracy_score(y_val, preds))