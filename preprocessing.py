import * from text_feature_preprocessing
import * from image_feature_preprocessing

X = Features('vqa')
X.import_features()
X.create_question_feature_df()
text_features_train, text_features_val = X.set_features(['QSN', 'descriptions', 'tags', 'dominant_colors','handwritten_text', 'ocr_text'])
X.set_targets()
embedding_matrix = X.get_word_embeddings()
train_seq = X.train_seq
val_seq   = X.val_seq

""" image features """
train_img, val_img = X.concat_image_features()

""" skill labels """
n_classes = 3
# check training set labels' class distribution
text_recognition_y_train = np.asarray(X.txt_train).astype('float32')
color_recognition_y_train = np.asarray(X.col_train).astype('float32')
counting_y_train = np.asarray(X.cnt_train).astype('float32')
print('Number of training samples each class: ')
print('Text recognition - 1: {} 0: {}'.format(np.count_nonzero(text_recognition_y_train), 
      len(text_recognition_y_train)-np.count_nonzero(text_recognition_y_train)))
print('Color recognition - 1:{} 0: {}'.format(np.count_nonzero(color_recognition_y_train),
      len(color_recognition_y_train)-np.count_nonzero(color_recognition_y_train)))
print('Counting - 1: {} 0: {}'.format(np.count_nonzero(counting_y_train), 
      len(counting_y_train)-np.count_nonzero(counting_y_train)))
# combine labels
y_train = np.column_stack((text_recognition_y_train, color_recognition_y_train, counting_y_train))
# check validation class distribution
text_recognition_y_val = np.asarray(X.txt_val).astype('float32')
color_recognition_y_val = np.asarray(X.col_val).astype('float32')
counting_y_val = np.asarray(X.cnt_val).astype('float32')
print('Number of validation samples each class: ')
print('Text recognition - 1: {} 0: {}'.format(np.count_nonzero(text_recognition_y_val), 
      len(text_recognition_y_val)-np.count_nonzero(text_recognition_y_val)))
print('Color recognition - 1:{} 0: {}'.format(np.count_nonzero(color_recognition_y_val),
     len(color_recognition_y_val)-np.count_nonzero(color_recognition_y_val)))
print('Counting - 1:{} 0: {}'.format(np.count_nonzero(counting_y_val),
     len(counting_y_val)-np.count_nonzero(counting_y_val)))
y_val = np.column_stack((text_recognition_y_val, color_recognition_y_val, counting_y_val))

# combine question and image features
X_train = np.vstack((train_seq, train_img))
X_val = np.vstack((val_seq, val_img))
val_data = (X_val, y_val)
