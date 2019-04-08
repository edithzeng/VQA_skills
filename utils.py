""" util """



VOCAB_SIZE = 50000
EMBEDDING_DIM = 300

def nn(MAX_DOC_LEN, train_seq, embedding_matrix,
	train_labels, val_data, learning_rate, lstm_dim, batch_size, 
	num_epochs, optimizer_param, regularization=1e-7, n_classes=3, verbose=0):
    l2_reg = regularizers.l2(regularization)
    # init model
    embedding_layer = Embedding(input_dim=VOCAB_SIZE,
                                output_dim=EMBEDDING_DIM,
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