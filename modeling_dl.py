#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:45:39 2019

@author: javadzabihi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

# Load the clean dataset
twitter_dataset = pd.read_csv('clean_tweeter_data.csv', index_col=0)
# twitter_dataset.head()

# Calculate the length of tweets
#twitter_dataset['clean_len'] = [len(t) for t in twitter_dataset.text]
# print('Minimum length of cleaned tweet:', twitter_dataset.clean_len.min())
# print('Maximum length of cleaned tweet:', twitter_dataset.clean_len.max())

# Create a boxplot of tweet length
# sns.set_style("whitegrid")
# fig, ax = plt.subplots(figsize=(8, 8))
# plt.boxplot(twitter_dataset.clean_len)
# plt.show()

y = twitter_dataset['target']
X = twitter_dataset['text']

# print('Shape of data:', X.shape)
# print('Shape of label:', y.shape)


# Train, Validation, and Test set : 98%, 2%, 2%
X_train, X, y_train, y = train_test_split(X, y, test_size=0.02, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, random_state=101)

# print ('Train set has {0} tweets with {1:.2f}% positive and {2:.2f}% negative sentiments'.format(len(X_train),
#    (len(X_train[y_train == 1]) / (len(X_train)*1.))*100,
#    (len(X_train[y_train == 0]) / (len(X_train)*1.))*100))

# print('Validation set has {0} tweets with {1:.2f}% positive and {2:.2f}% negative sentiments'.format(len(X_val),
#    (len(X_val[y_val == 1]) / (len(X_val)*1.))*100,
#   (len(X_val[y_val == 0]) / (len(X_val)*1.))*100 ))

# print('Test set has {0} tweets with {1:.2f}% positive and {2:.2f}% negative sentiments'.format(len(X_test),
#    (len(X_test[y_test == 1]) / (len(X_test)*1.))*100,
#    (len(X_test[y_test == 0]) / (len(X_test)*1.))*100 ))


# Number of words to consider as features
max_features = 100000
# Cuts off the text after this number of words
maxlen = 60
# Create a tokenizer, configured to only take into account the 100,000 most common words
tokenizer = Tokenizer(num_words=max_features)
# Build the word index
tokenizer.fit_on_texts(twitter_dataset['text'].values)
# Turn strings into lists of integer indices
train_sequences = tokenizer.texts_to_sequences(X_train)
# All sequences in abatch must have the same length
# Sequences with length less than maxlen should be padded with zeros
# Sequences with length longer than maxlen should be truncated
X_train_sequences = pad_sequences(train_sequences, maxlen=maxlen)

# Validation set
val_sequences = tokenizer.texts_to_sequences(X_val)
X_val_sequences = pad_sequences(val_sequences, maxlen=maxlen)

# Test set
test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_sequences = pad_sequences(test_sequences, maxlen=maxlen)

#from tqdm import tqdm
# tqdm.pandas(desc="progress-bar")
#import gensim
#from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
#from sklearn import utils

cores = multiprocessing.cpu_count()


def labelize_tweets_ug(tweets, label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


# Combine X_train, X_val, X_test
all_x = pd.concat([X_train, X_val, X_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')


from gensim.models import word2vec


# CBOW

model_ug_cbow = Word2Vec( workers = cores, # Number of parallel threads
                         size = 100, # Word vector dimensionality
                         min_count= 2, # Minimum word count
                         window= 5, # Context window size
                         alpha = 0.07, # The initial learning rate
                         min_alpha = 0.07) # Learning rate will linearly drop to `min_alpha` as training progresses

model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])


# train model_ug_cbow
for epoch in range(20):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.001
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

# skip-gram

model_ug_sg = Word2Vec( workers = cores, # Number of parallel threads
                        size = 100, # Word vector dimensionality
                        min_count= 2, # Minimum word count
                        window= 5, # Context window size
                        alpha = 0.07, # The initial learning rate
                        min_alpha = 0.07, # Learning rate will linearly drop to `min_alpha` as training progresses
                        sg = 1) # skip-gram
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

# # train model_ug_sg
for epoch in range(20):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.001
    model_ug_sg.min_alpha = model_ug_sg.alpha

# Save models
model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
model_ug_sg.save('w2v_model_ug_sg.word2vec')

from gensim.models import KeyedVectors
model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')

# Create a dictionary to extract word vectors from
# concatenate vectors of the two models
# Then each word will have 400 dimension
embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w], model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))

# create a matrix of word vectors
embedding_matrix = np.zeros((max_features, 200))
for word, i in tokenizer.word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


# EarlyStopping:
# Interrupts training when improvement stops
# Monitors the model’s validation accuracy
# Interrupts training when accuracy has stopped improving
# for more than two epoch (that is, three epochs)
# EarlyStopping(monitor='acc', patience=2)

# ModelCheckpoint:
# Saves the current weights after every epoch
# you won’t overwrite the model file unless val_loss has improved,
# which allows you to keep the best model seen during training.

# ReduceLROnPlateau:
# Reduce learning rate when the validation loss stopped improving
# Divides the learning rate by 10 when triggered
# The callback is triggered after the validation loss has stopped improving for 3 epochs.

# TensorBoard:
# Visually monitor metrics, histograms of activations and gradients during training
# Records activation histograms every 1 epoch
# Records embedding data every 1 epoch
# TensorBoard(histogram_freq=1, embeddings_freq=1)

def callbacks_list(model_name):
    callbacks_list = [ModelCheckpoint(filepath='%s.h5' % model_name,
                                      monitor='val_loss',
                                      save_best_only=True),
                      ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        patience=3)]

    return callbacks_list


from keras.optimizers import adam


# compile the model
def compile_model(model, optimizer=adam(lr=0.001),
                  loss='binary_crossentropy', metrics=['acc']):
    return model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)


# fit on the training set
def fit_model(model, callbacks, X_train=X_train_sequences, y_train=y_train,
              epochs=6, batch_size=256,
              validation_data=(X_val_sequences, y_val)):
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks)

    return history


# save history of the model in a csv file
def save_history(model_history, model_name):
    history = pd.DataFrame(model_history.history, columns=model_history.history.keys())
    history.to_csv('%s_history.csv' % model_name)

# Load the best model
def load_best_model(model_name, x=X_val_sequences, y=y_val):
    best_model = load_model('%s.h5' % model_name)
    Loss, Accuracy = best_model.evaluate(x, y)
    print('Accuracy (%s): %f' % (model_name, Accuracy * 100))
    return best_model


# plot the learning curve
def learning_curve(model_name):

    plt.figure(figsize=(8, 4))
    sns.set(font_scale=1)
    sns.set_style('whitegrid')
    history = pd.read_csv('%s_history.csv'% model_name)
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, '-', label='Training acc')
    plt.plot(epochs, val_acc, '--', label='Validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, '-', label='Training loss')
    plt.plot(epochs, val_loss, '--', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('learning_curve_%s.png'% model_name)
    plt.show()


# ROC Curve
# model_names: List of model names
def roc_plot(model_names, x_test=X_test_sequences, y_test=y_test):

    plt.figure(figsize=(8, 4))
    sns.set(font_scale=1)
    sns.set_style('whitegrid')

    for model in model_names:

        best_model = load_model('%s.h5'% model)
        y_pred_probs = best_model.predict(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (model ,roc_auc), linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Characteristic')
        plt.legend(loc=0)
        plt.tight_layout()
    plt.savefig('roc_%s.png' % (model_names))
    plt.show()


# Performance Table + ROC Curve
# model_names: List of model names
def performance_results(model_names, x_test=X_test_sequences, y_test=y_test):
    acc_dict = {}
    auc_dict = {}
    precision_dict = {}
    recall_dict = {}

    for model in model_names:

        best_model = load_model('%s.h5' % model)
        loss, acc = best_model.evaluate(x_test, y_test)
        acc_dict[model] = round(acc * 100, 2)
        y_pred_probs = np.asarray(best_model.predict(x_test))
        fpr, tpr, threshold = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        auc_dict[model] = round(roc_auc, 3)
        if 'inception' in model:
            y_pred_classes = (y_pred_probs>0.5)*1
        else:
            y_pred_classes = best_model.predict_classes(x_test)
        precision = precision_score(y_test, y_pred_classes)
        precision_dict[model] = round(precision * 100, 2)
        recall = recall_score(y_test, y_pred_classes)
        recall_dict[model] = round(recall * 100, 2)
        # roc curve
        plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (model, roc_auc), linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        plt.legend(loc=0)
        plt.tight_layout()
    plt.savefig('roc_%s.png' % model_names)
    plt.show()

    acc_data = pd.DataFrame(acc_dict.items(), columns = ['Model', 'Accuracy'])
    auc_data = pd.DataFrame(auc_dict.items(), columns=['Model', 'AUC'])
    precision_data = pd.DataFrame(precision_dict.items(), columns=['Model', 'Precision'])
    recall_data = pd.DataFrame(recall_dict.items(), columns=['Model', 'Recall'])
    acc_auc = pd.merge(acc_data, auc_data, on = 'Model')
    precision_recall = pd.merge(precision_data, recall_data, on = 'Model')
    performance_table = pd.merge(acc_auc, precision_recall, on = 'Model').set_index('Model')
    # save in a csv file
    performance_table.to_csv('performance_%s.csv'% model_names)
    return performance_table




# Baseline Model
# Logistic Regression + TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



def baseline_model(max_features=100000,ngram_range=(1, 3),
                   X_train = X_train, y_train=y_train, X_test = X_test, y_test = y_test):
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range)
    tfidf.fit(X_train)
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    # Logistic Regression
    lr_tfidf = LogisticRegression()
    lr_tfidf.fit(X_train_tfidf,y_train)
    accuracy = lr_tfidf.score(X_test_tfidf,y_test)
    accuracy = round(accuracy * 100, 2)
    y_pred_probs = lr_tfidf.predict_proba(X_test_tfidf)
    y_pred_probs = y_pred_probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    y_pred_classes = lr_tfidf.predict(X_test_tfidf)
    precision = precision_score(y_test, y_pred_classes)
    precision = round(precision * 100, 2)
    recall = recall_score(y_test, y_pred_classes)
    recall = round(recall * 100, 2)
    baseline_performance = pd.DataFrame([{'Model':'logistic_regression_tfidf',
                                          'Accuracy':accuracy, 'AUC':roc_auc,
                                          'Precision':precision, 'Recall':recall}]).set_index('Model')
    # save to a csv file
    baseline_performance.to_csv('performance_baseline_model.csv')
    return baseline_performance


from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras import regularizers
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from keras.models import load_model
from keras.layers import Reshape
from keras.layers import Input, concatenate
from keras.models import Model
from keras.layers import Bidirectional



# Feed-Forward
# embed_dim : Embedding dimensionality
# A simple Neural Net (Embedding + Flatten + Dropout + Dense)
# Use pre-trained word embedding. Set the weights attribute equal to our embedding_matrix.
# we do not want to update the learned word weights in this model,
# therefore we will set the trainable attribute for the model to False.
def feedforward_w2v(max_features=100000, embed_dim=200, dropout=0.5,
                    embedding_matrix=embedding_matrix, output_dense_dim=128):

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))
    # Flattens the 3D tensor of embeddings into a
    # 2D tensor of shape (samples, maxlen * embed_dim)
    model.add(Flatten())
    # First dense layer
    model.add(Dense(output_dense_dim, activation='relu'))
    # dropout
    model.add(Dropout(dropout))
    # Adds the classifier on top
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


# Feed-Forward  (fixed pre-trained word2vec)
feedforward_w2v_model = feedforward_w2v(dropout = 0.2, output_dense_dim = 128)
# compile the model
compile_model(feedforward_w2v_model)
# fit on the training set
feedforward_w2v_history = fit_model(feedforward_w2v_model, callbacks_list('feedforward_w2v'), batch_size=256)
# Save history
save_history(feedforward_w2v_history, 'feedforward_w2v')
# learning curve
learning_curve('feedforward_w2v')
# Load the best model
best_feedforward_w2v = load_best_model('feedforward_w2v')


# Stacked Feed-Forward
# embed_dim : Embedding dimensionality
# Use pre-trained word embedding. Set the weights attribute equal to our embedding_matrix.
# we do not want to update the learned word weights in this model,
# therefore we will set the trainable attribute for the model to False.
def stacked_feedforward_w2v(max_features=100000, embed_dim=200,
                            embedding_matrix=embedding_matrix,
                            output_dense_dim1=128, output_dense_dim2=128, output_dense_dim3=128,
                            dropout1=0.2, dropout2=0.2, dropout3=0.2):

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))
    # Flattens the 3D tensor of embeddings into a
    # 2D tensor of shape (samples, maxlen * embed_dim)
    model.add(Flatten())
    # First dense layer
    model.add(Dense(output_dense_dim1, activation='relu'))
    # first dropout
    model.add(Dropout(dropout1))
    # second dense layer
    model.add(Dense(output_dense_dim2, activation='relu'))
    # second dropout
    model.add(Dropout(dropout2))
    # third dense layer
    model.add(Dense(output_dense_dim3, activation='relu'))
    # third dropout
    model.add(Dropout(dropout3))
    # Adds the classifier on top
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


# Stacked Feed-Forward  (fixed pre-trained word2vec)
stacked_feedforward_w2v_model = stacked_feedforward_w2v()
# compile the model
compile_model(stacked_feedforward_w2v_model)
# fit on the training set
stacked_feedforward_w2v_history = fit_model(stacked_feedforward_w2v_model,
                                            callbacks_list('stacked_feedforward_w2v'), batch_size=256)
# Save history
save_history(stacked_feedforward_w2v_history, 'stacked_feedforward_w2v')
# learning curve
learning_curve('stacked_feedforward_w2v')
# Load the best model
best_stacked_feedforward_w2v = load_best_model('stacked_feedforward_w2v')



# CNN
# Fixed pre-trained w2v
# nb_filter : Number of filters
# conv_window : Convolution window size
def cnn_w2v(max_features=100000, embed_dim=200, dropout=0.2, nb_filter=128, conv_window=3,
            output_dense_dim=256, embedding_matrix=embedding_matrix):

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropout))

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model

# CNN (fixed pre-trained word2vec)
cnn_w2v_model = cnn_w2v(dropout=0.2, output_dense_dim=256)
# compile the model
compile_model(cnn_w2v_model)
# fit on the training set
cnn_w2v_history = fit_model(cnn_w2v_model, callbacks_list('cnn_w2v'), batch_size=256)
# Save history
save_history(cnn_w2v_history, 'cnn_w2v')
# learning curve
learning_curve ('cnn_w2v')
# Load the best model
best_cnn_w2v = load_best_model('cnn_w2v')




# Stacked CNN
# nb_filter : Number of filters
# conv_window : Convolution window size
def stacked_cnn_w2v(max_features=100000, embed_dim=200, dropout=0.2, nb_filter=128, conv_window=3,
                    pool_size=2, output_dense_dim=256, embedding_matrix=embedding_matrix):

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(MaxPooling1D(pool_size))

    model.add(Dropout(dropout))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropout))

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


# Stacked CNN (fixed pre-trained word2vec)
stacked_cnn_w2v_model = stacked_cnn_w2v(dropout=0.2, output_dense_dim=256)
# compile the model
compile_model(stacked_cnn_w2v_model)
# fit on the training set
stacked_cnn_w2v_history = fit_model(stacked_cnn_w2v_model, callbacks_list('stacked_cnn_w2v'), batch_size=256)
# Save history
save_history(stacked_cnn_w2v_history, 'stacked_cnn_w2v')
# learning curve
learning_curve ('stacked_cnn_w2v')
# Load the best model
best_stacked_cnn_w2v = load_best_model('stacked_cnn_w2v')



# LSTM
# A simple LSTM (Embedding + LSTM + Dense)
# frozen embedding layer (fixed pre-trained word2vec)
# LSTM_out : The LSTM output dimensionality
# embed_dim: Embedding dimensionality
def lstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, recurrent_dropout=0.2,
              lstm_out=128, output_dense_dim=256, embedding_matrix=embedding_matrix):

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model

# LSTM  (fixed pre-trained word2vec)
lstm_w2v_model = lstm_w2v(dropout = 0.2, output_dense_dim = 256)
# compile the model
compile_model(lstm_w2v_model)
# fit on the training set
lstm_w2v_history = fit_model(lstm_w2v_model, callbacks_list('lstm_w2v'), batch_size=256)
# Save history
save_history(lstm_w2v_history, 'lstm_w2v')
# learning curve
learning_curve('lstm_w2v')
# Load the best model
best_lstm_w2v = load_best_model('lstm_w2v')


# Stacked LSTM
# To stack recurrent layers on top of each other, we need to specify return_sequences=True
# for all intermediate layers to return their full sequence of outputs
# rather than their output at the last timestep.
def stacked_lstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, recurrent_dropout=0.2,
            output_dense_dim=256, embedding_matrix=embedding_matrix,
            lstm_out_1=128, lstm_out_2=128, lstm_out_3=128):

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(LSTM(lstm_out_1,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout,
                   return_sequences=True))

    model.add(LSTM(lstm_out_2,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout,
                   return_sequences=True))

    model.add(LSTM(lstm_out_3,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout))

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model

# Stacked LSTM  (fixed pre-trained word2vec)
stacked_lstm_w2v_model = stacked_lstm_w2v(dropout = 0.2, output_dense_dim = 256)
# compile the model
compile_model(stacked_lstm_w2v_model)
# fit on the training set
stacked_lstm_w2v_history = fit_model(stacked_lstm_w2v_model, callbacks_list('stacked_lstm_w2v'), batch_size=256)
# Save history
save_history(stacked_lstm_w2v_history, 'stacked_lstm_w2v')
# learning curve
learning_curve('stacked_lstm_w2v')
# Load the best model
best_stacked_lstm_w2v = load_best_model('stacked_lstm_w2v')


# Bidirectional LSTM
# Bidirectional creates a second, separate instance for the recurrent layer and it uses
# one instance in chronological order and the other one in the reversed order.
def bilstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, recurrent_dropout=0.2,
               lstm_out=128, output_dense_dim=256, embedding_matrix=embedding_matrix):

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(Bidirectional(LSTM(lstm_out,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)))

    model.add(Flatten())

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


# Bidirectional LSTM (fixed pre-trained word2vec)
bilstm_w2v_model = bilstm_w2v(dropout = 0.2, recurrent_dropout = 0.2, output_dense_dim = 256)
# compile the model
compile_model(bilstm_w2v_model)
# fit on the training set
bilstm_w2v_history = fit_model(bilstm_w2v_model, callbacks_list('bilstm_w2v'), batch_size=256)
# Save history
save_history(bilstm_w2v_history, 'bilstm_w2v')
# learning curve
learning_curve('bilstm_w2v')
# Load the best model
best_bilstm_w2v = load_best_model('bilstm_w2v')


# Stacked Bidirectional LSTM
# Bidirectional creates a second, separate instance for the recurrent layer and it uses
# one instance in chronological order and the other one in the reversed order.
def stacked_bilstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, recurrent_dropout=0.2,
            output_dense_dim=256, embedding_matrix=embedding_matrix,
            lstm_out_1=128, lstm_out_2=128, lstm_out_3 = 128):

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(Bidirectional(LSTM(lstm_out_1,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)))

    model.add(Bidirectional(LSTM(lstm_out_2,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)))

    model.add(Bidirectional(LSTM(lstm_out_3,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)))

    model.add(Flatten())

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model

# Stacked Bidirectional LSTM (fixed pre-trained word2vec)
stacked_bilstm_w2v_model = stacked_bilstm_w2v(dropout = 0.2, recurrent_dropout = 0.2, output_dense_dim = 256)
# compile the model
compile_model(stacked_bilstm_w2v_model)
# fit on the training set
stacked_bilstm_w2v_history = fit_model(stacked_bilstm_w2v_model, callbacks_list('stacked_bilstm_w2v'), batch_size=256)
# Save history
save_history(stacked_bilstm_w2v_history, 'stacked_bilstm_w2v')
# learning curve
learning_curve('stacked_bilstm_w2v')
# Load the best model
best_stacked_bilstm_w2v = load_best_model('stacked_bilstm_w2v')



# CNN + LSTM
def cnn_lstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, nb_filter=128, conv_window=3,
              pool_size=2, output_dense_dim=256, embedding_matrix=embedding_matrix,
              recurrent_dropout=0.2, lstm_out=128):
    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(MaxPooling1D(pool_size))

    model.add(Dropout(dropout))

    model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model

# CNN + LSTM
cnn_lstm_w2v_model = cnn_lstm_w2v(dropout=0.2, output_dense_dim=256)
# compile the model
compile_model(cnn_lstm_w2v_model)
# fit on the training set
cnn_lstm_w2v_history = fit_model(cnn_lstm_w2v_model, callbacks_list('cnn_lstm_w2v'), batch_size=256)
# Save history
save_history(cnn_lstm_w2v_history, 'cnn_lstm_w2v')
# learning curve
learning_curve ('cnn_lstm_w2v')
# Load the best model
best_cnn_lstm_w2v = load_best_model('cnn_lstm_w2v')


# LSTM + CNN (fixed pre-trained word2vec)
def lstm_cnn_w2v(max_features=100000, embed_dim=200, dropout=0.2, nb_filter=128, conv_window=3,
               pool_size=2, output_dense_dim=256, embedding_matrix=embedding_matrix,
               recurrent_dropout=0.2, lstm_out=128):
    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(Reshape((lstm_out, 1)))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(MaxPooling1D(pool_size))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(Flatten())

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


# LSTM + CNN (fixed pre-trained word2vec)
lstm_cnn_w2v_model = lstm_cnn_w2v(dropout = 0.2, output_dense_dim = 256)
# compile the model
compile_model(lstm_cnn_w2v_model)
# fit on the training set
lstm_cnn_w2v_history = fit_model(lstm_cnn_w2v_model, callbacks_list('lstm_cnn_w2v'), batch_size=256)
# Save history
save_history(lstm_cnn_w2v_history, 'lstm_cnn_w2v')
# learning curve
learning_curve ('lstm_cnn_w2v')
# Load the best model
best_lstm_cnn_w2v = load_best_model('lstm_cnn_w2v')



# CNN + Bidirectional LSTM
def cnn_bilstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, nb_filter=128, conv_window=3,
              pool_size=2, output_dense_dim=256, embedding_matrix=embedding_matrix,
              recurrent_dropout=0.2, lstm_out=128):

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(MaxPooling1D(pool_size))

    model.add(Dropout(dropout))

    model.add(Bidirectional(LSTM(lstm_out,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)))

    model.add(Flatten())

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


# CNN + bidirectional LSTM
cnn_bilstm_w2v_model = cnn_bilstm_w2v(dropout=0.2, output_dense_dim=256)
# compile the model
compile_model(cnn_bilstm_w2v_model)
# fit on the training set
cnn_bilstm_w2v_history = fit_model(cnn_bilstm_w2v_model, callbacks_list('cnn_bilstm_w2v'), batch_size=256)
# Save history
save_history(cnn_bilstm_w2v_history, 'cnn_bilstm_w2v')
# learning curve
learning_curve('cnn_bilstm_w2v')
# Load the best model
best_cnn_bilstm_w2v = load_best_model('cnn_bilstm_w2v')


# Bidirectional LSTM + CNN (fixed pre-trained word2vec)
def bilstm_cnn_w2v(max_features=100000, embed_dim=200, dropout=0.2, nb_filter=128, conv_window=3,
               pool_size=2, output_dense_dim=256, embedding_matrix=embedding_matrix,
               recurrent_dropout=0.2, lstm_out=128):
    model = Sequential()

    model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix],
                        input_length=maxlen, trainable=False))

    model.add(Bidirectional(LSTM(lstm_out,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(MaxPooling1D(pool_size))

    model.add(Conv1D(nb_filter, conv_window, activation='relu'))

    model.add(Flatten())

    model.add(Dense(output_dense_dim, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    return model


# Bidirectional LSTM + CNN (fixed pre-trained word2vec)
bilstm_cnn_w2v_model = bilstm_cnn_w2v(dropout = 0.2, output_dense_dim = 256)
# compile the model
compile_model(bilstm_cnn_w2v_model)
# fit on the training set
bilstm_cnn_w2v_history = fit_model(bilstm_cnn_w2v_model, callbacks_list('bilstm_cnn_w2v'), batch_size=256)
# Save history
save_history(bilstm_cnn_w2v_history, 'bilstm_cnn_w2v')
# learning curve
learning_curve('bilstm_cnn_w2v')
# Load the best model
best_bilstm_cnn_w2v = load_best_model('bilstm_cnn_w2v')



# An Inception Module with CNN
def inception_cnn_w2v(max_features=100000, embed_dim=200, dropout=0.2, output_dense_dim=256,
             embedding_matrix=embedding_matrix, nb_filter_a=128, conv_window_a=3,
             nb_filter_b=128, conv_window_b=4,
             nb_filter_c=128, conv_window_c=5):

    tweets_input = Input(shape=(maxlen,), dtype='int32')

    embed_layer = Embedding(max_features, embed_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=False)(tweets_input)

    branch_a = Conv1D(nb_filter_a, conv_window_a, activation='relu')(embed_layer)
    branch_a = GlobalMaxPooling1D()(branch_a)


    branch_b = Conv1D(nb_filter_b, conv_window_b, activation='relu')(embed_layer)
    branch_b = GlobalMaxPooling1D()(branch_b)


    branch_c = Conv1D(nb_filter_c, conv_window_c, activation='relu')(embed_layer)
    branch_c = GlobalMaxPooling1D()(branch_c)


    output = concatenate([branch_a, branch_b, branch_c], axis=1)
    output = Dropout(dropout)(output)
    output = Dense(output_dense_dim, activation='relu')(output)
    output = Dropout(dropout)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[tweets_input], outputs=[output])

    print(model.summary())

    return model


# An Inception Module with CNN (fixed pre-trained word2vec)
inception_cnn_w2v_model = inception_cnn_w2v(dropout = 0.2, output_dense_dim = 256)
# compile the model
compile_model(inception_cnn_w2v_model)
# fit on the training set
inception_cnn_w2v_history = fit_model(inception_cnn_w2v_model, callbacks_list('inception_cnn_w2v'), batch_size=256)
# Save history
save_history(inception_cnn_w2v_history ,'inception_cnn_w2v')
# learning curve
learning_curve('inception_cnn_w2v')
# Load the best model
best_inception_cnn_w2v = load_best_model('inception_cnn_w2v')


# An Inception Module with LSTM
def inception_lstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, output_dense_dim=128,
             embedding_matrix=embedding_matrix, lstm_out_a = 128, lstm_out_b = 128, lstm_out_c = 128,
             dropout_a=0.2, dropout_b=0.3, dropout_c=0.4,
             recurrent_dropout_a=0.2, recurrent_dropout_b=0.3, recurrent_dropout_c=0.4):

    tweets_input = Input(shape=(maxlen,), dtype='int32')

    embed_layer = Embedding(max_features, embed_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=False)(tweets_input)

    branch_a = LSTM(lstm_out_a, dropout=dropout_a, recurrent_dropout=recurrent_dropout_a)(embed_layer)

    branch_b = LSTM(lstm_out_b, dropout=dropout_b, recurrent_dropout=recurrent_dropout_b)(embed_layer)

    branch_c = LSTM(lstm_out_c, dropout=dropout_c, recurrent_dropout=recurrent_dropout_c)(embed_layer)

    output = concatenate([branch_a, branch_b, branch_c], axis=1)
    output = Dense(output_dense_dim, activation='relu')(output)
    output = Dropout(dropout)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[tweets_input], outputs=[output])

    print(model.summary())

    return model

# An Inception Module for LSTM (fixed pre-trained word2vec)
inception_lstm_w2v_model = inception_lstm_w2v(dropout = 0.2, output_dense_dim = 128)
# compile the model
compile_model(inception_lstm_w2v_model)
# fit on the training set
inception_lstm_w2v_history = fit_model(inception_lstm_w2v_model, callbacks_list('inception_lstm_w2v'), batch_size=256)
# Save history
save_history(inception_lstm_w2v_history, 'inception_lstm_w2v')
#learning curve
learning_curve('inception_lstm_w2v')
# Load the best model
best_inception_lstm_w2v = load_best_model('inception_lstm_w2v')



### An Inception Module with Bidirectional LSTM
def inception_bilstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, output_dense_dim=128,
                         embedding_matrix=embedding_matrix,
                         lstm_out_a=128, recurrent_dropout_a=0.2, dropout_a=0.2,
                         lstm_out_b=128, recurrent_dropout_b=0.2, dropout_b=0.2,
                         lstm_out_c=128, dropout_c=0.2, recurrent_dropout_c=0.2):

    tweets_input = Input(shape=(maxlen,), dtype='int32')

    embed_layer = Embedding(max_features, embed_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=False)(tweets_input)

    branch_a = Bidirectional(LSTM(lstm_out_a,
                                  dropout=dropout_a,
                                  recurrent_dropout=recurrent_dropout_a,
                                  return_sequences=True))(embed_layer)
    branch_a = Flatten()(branch_a)
    branch_a = Dense(output_dense_dim, activation='relu')(branch_a)

    branch_b = Bidirectional(LSTM(lstm_out_b,
                                  dropout=dropout_b,
                                  recurrent_dropout=recurrent_dropout_b,
                                  return_sequences=True))(embed_layer)
    branch_b = Flatten()(branch_b)
    branch_b = Dense(output_dense_dim, activation='relu')(branch_b)

    branch_c = Bidirectional(LSTM(lstm_out_c,
                                  dropout=dropout_c,
                                  recurrent_dropout=recurrent_dropout_c,
                                  return_sequences=True))(embed_layer)
    branch_c = Flatten()(branch_c)
    branch_c = Dense(output_dense_dim, activation='relu')(branch_c)

    output = concatenate([branch_a, branch_b, branch_c], axis=1)
    output = Dense(output_dense_dim, activation='relu')(output)
    output = Dropout(dropout)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[tweets_input], outputs=[output])

    print(model.summary())

    return model


# An Inception Module with Bidirectional LSTM (fixed pre-trained word2vec)
inception_bilstm_w2v_model = inception_bilstm_w2v(output_dense_dim = 128)
# compile the model
compile_model(inception_bilstm_w2v_model)
# fit on the training set
inception_bilstm_w2v_history = fit_model(inception_bilstm_w2v_model,
                                         callbacks_list('inception_bilstm_w2v'),
                                         batch_size=256)
# Save history
save_history(inception_bilstm_w2v_history, 'inception_bilstm_w2v')
# learning curve
learning_curve('inception_bilstm_w2v')
# Load the best model
best_inception_bilstm_w2v = load_best_model('inception_bilstm_w2v')



### An Inception Module with LSTM, CNN, Bidirectional LSTM
def inception_cnn_lstm_bilstm_w2v(max_features=100000, embed_dim=200, dropout=0.2, output_dense_dim=128,
             embedding_matrix=embedding_matrix,
             lstm_out_a=128, recurrent_dropout_a=0.2, dropout_a=0.2, nb_filter_b=128, conv_window_b=3,
             lstm_out_c=128, dropout_c=0.2, recurrent_dropout_c=0.2):

    tweets_input = Input(shape=(maxlen,), dtype='int32')

    embed_layer = Embedding(max_features, embed_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=False)(tweets_input)

    branch_a = LSTM(lstm_out_a, dropout=dropout_a, recurrent_dropout=recurrent_dropout_a)(embed_layer)

    branch_b = Conv1D(nb_filter_b, conv_window_b, activation='relu')(embed_layer)
    branch_b = GlobalMaxPooling1D()(branch_b)

    branch_c = Bidirectional(LSTM(lstm_out_c,
                                  dropout=dropout_c,
                                  recurrent_dropout=recurrent_dropout_c,
                                  return_sequences=True))(embed_layer)
    branch_c = Flatten()(branch_c)
    branch_c = Dense(output_dense_dim, activation='relu')(branch_c)

    output = concatenate([branch_a, branch_b, branch_c], axis=1)
    output = Dense(output_dense_dim, activation='relu')(output)
    output = Dropout(dropout)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[tweets_input], outputs=[output])

    print(model.summary())

    return model


# An Inception Module with LSTM, CNN, Bidirectional LSTM (fixed pre-trained word2vec)
inception_cnn_lstm_bilstm_w2v_model = inception_cnn_lstm_bilstm_w2v(output_dense_dim = 128)
# compile the model
compile_model(inception_cnn_lstm_bilstm_w2v_model)
# fit on the training set
inception_cnn_lstm_bilstm_w2v_history = fit_model(inception_cnn_lstm_bilstm_w2v_model,
                                                  callbacks_list('inception_cnn_lstm_bilstm_w2v'), 
                                                  batch_size=256)
# Save history
save_history(inception_cnn_lstm_bilstm_w2v_history, 'inception_cnn_lstm_bilstm_w2v')
# learning curve
learning_curve('inception_cnn_lstm_bilstm_w2v')
# Load the best model
best_inception_cnn_lstm_bilstm_w2v = load_best_model('inception_cnn_lstm_bilstm_w2v')

#### Results (Plots, Tables)
#feedforward_performance = performance_results(['feedforward_w2v', 'stacked_feedforward_w2v'])
cnn_lstm_performance = performance_results(['cnn_w2v', 'stacked_cnn_w2v', 'lstm_w2v', 'stacked_lstm_w2v'])
bilstm_lstm_performance = performance_results(['bilstm_w2v', 'stacked_bilstm_w2v', 'lstm_w2v', 'stacked_lstm_w2v'])
#cnnlstm_performance = performance_results(['cnn_lstm_w2v', 'lstm_cnn_w2v'])
#inception_cnn_lstm_performance = performance_results(['inception_lstm_w2v', 'inception_cnn_w2v'])
#inception_cnnlstm_performance = performance_results(['cnn_lstm_w2v', 'lstm_cnn_w2v', 'inception_lstm_w2v', 'inception_cnn_w2v'])
#inception_performance = performance_results(['inception_lstm_w2v', 'inception_cnn_w2v', 'inception_cnn_lstm_bilstm_w2v'])
lstm_performance = performance_results(['lstm_w2v', 'stacked_lstm_w2v', 'inception_lstm_w2v'])
cnn_performance = performance_results(['cnn_w2v', 'stacked_cnn_w2v', 'inception_cnn_w2v'])
bilstm_performance = performance_results(['bilstm_w2v', 'stacked_bilstm_w2v', 'inception_bilstm_w2v'])
lstm_cnn_bilstm_performance = performance_results(['cnn_lstm_w2v', 'lstm_cnn_w2v', 'cnn_bilstm_w2v', 'bilstm_cnn_w2v'])
inception_performance = performance_results(['inception_lstm_w2v', 'inception_bilstm_w2v', 'inception_cnn_w2v', 'inception_cnn_lstm_bilstm_w2v'])

# Baseline performance
baseline_performance = baseline_model()
# Concatenate two performance tables (Final Models, Baseline Model)
final_performance_table = pd.concat([baseline_performance, cnn_lstm_performance,
                                     bilstm_lstm_performance, lstm_performance,
                                     cnn_performance, bilstm_performance,
                                     lstm_cnn_bilstm_performance, inception_performance], axis=0)

# drop duplicates and save the result to a csv file
final_performance_table = final_performance_table.reset_index().drop_duplicates(subset='Model', keep='last').set_index('Model')
final_performance_table.to_csv('final_performance_table.csv')

