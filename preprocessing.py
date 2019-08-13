from extract_features import *
from utils import *

def preprocess(dataset, features=['QSN','descriptions','tags','dominant_colors','handwritten_text','ocr_text'], n_classes=3, skill=None, verbose=True):

	if not isinstance(features, list):
		raise ValueError("Check features list")
	print("Dataset:", dataset, "\nFeatures:", features)

	X = Features(dataset)
	X.import_features()
	X.create_question_feature_df()
	X.set_features(features)
	X.set_targets()
	embedding_matrix = X.get_word_embedding()
	train_seq = X.train_seq
	val_seq   = X.val_seq

	""" image features """
	# train_img, val_img = X.concat_image_features()
	# check training set labels' class distribution
	text_recognition_y_train = np.asarray(X.txt_train).astype('float32')
	color_recognition_y_train = np.asarray(X.col_train).astype('float32')
	counting_y_train = np.asarray(X.cnt_train).astype('float32')
	train_dict = {"text": text_recognition_y_train, "color": color_recognition_y_train, 'counting': counting_y_train}
	if verbose:
		print('Number of training samples each class: ')
		print('Text recognition - 1: {} 0: {}'.format(np.count_nonzero(text_recognition_y_train), 
	      len(text_recognition_y_train)-np.count_nonzero(text_recognition_y_train)))
		print('Color recognition - 1:{} 0: {}'.format(np.count_nonzero(color_recognition_y_train),
	      len(color_recognition_y_train)-np.count_nonzero(color_recognition_y_train)))
		print('Counting - 1: {} 0: {}'.format(np.count_nonzero(counting_y_train), 
	      len(counting_y_train)-np.count_nonzero(counting_y_train)))

	# combine labels
	if n_classes == 1:
		y_train = train_dict[skill]
	elif n_classes == 2:
		y_train = np.column_stack((text_recognition_y_train, color_recognition_y_train))
	elif n_classes == 3:
		y_train = np.column_stack((text_recognition_y_train, color_recognition_y_train, counting_y_train))
	# check validation class distribution
	text_recognition_y_val = np.asarray(X.txt_val).astype('float32')
	color_recognition_y_val = np.asarray(X.col_val).astype('float32')
	counting_y_val = np.asarray(X.cnt_val).astype('float32')
	val_dict = {"text": text_recognition_y_val, "color": color_recognition_y_val, 'counting': counting_y_val}
	if verbose:
		print('Number of validation samples each class: ')
		print('Text recognition - 1: {} 0: {}'.format(np.count_nonzero(text_recognition_y_val), 
	      len(text_recognition_y_val)-np.count_nonzero(text_recognition_y_val)))
		print('Color recognition - 1:{} 0: {}'.format(np.count_nonzero(color_recognition_y_val),
	     len(color_recognition_y_val)-np.count_nonzero(color_recognition_y_val)))
		print('Counting - 1:{} 0: {}'.format(np.count_nonzero(counting_y_val),
	     len(counting_y_val)-np.count_nonzero(counting_y_val)))
	if n_classes == 1:
		y_val = val_dict[skill]
	elif n_classes == 2:
		y_val = np.column_stack((text_recognition_y_val, color_recognition_y_val)) 
	elif n_classes == 3:
		y_val = np.column_stack((text_recognition_y_val, color_recognition_y_val, counting_y_val))

	# PCA to reduce dimensionality
	MAX_DOC_LEN = 40
	if verbose:
		print("PCA with {} eigenvectors".format(MAX_DOC_LEN))
	train_seq, val_seq = preprocess_pca(train_seq, val_seq, dim=MAX_DOC_LEN)

	return embedding_matrix, train_seq, val_seq, y_train, y_val
