#!/usr/bin/env python
# coding=utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from keras import backend

# To have reproducability: Set all the seeds, make sure multithreading is off, if possible don't use GPU. 
tf.set_random_seed(1)
np.random.seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))
pd.set_option('display.max_colwidth', -1)


train_raw = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', delimiter="\t").fillna('')
test_raw = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', delimiter="\t").fillna('')

print('Input Files read')

NUM_FOLDS = 5

train_raw["fold_id"] = train_raw["SentenceId"].apply(lambda x: x%NUM_FOLDS)

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


vocab_size = 20000  # based on words in the entire corpus
max_len = 60        # based on word count in phrases

all_corpus   = list(train_raw['Phrase'].values) + list(test_raw['Phrase'].values)
train_phrases  = list(train_raw['Phrase'].values) 
test_phrases   = list(test_raw['Phrase'].values)
X_train_target_binary = pd.get_dummies(train_raw['Sentiment'])

#Vocabulary-Indexing of thetrain and test phrases, make sure "filters" parm doesn't clean out punctuations

tokenizer = Tokenizer(num_words=vocab_size, lower=True, filters='\n\t')
tokenizer.fit_on_texts(all_corpus)
encoded_train_phrases = tokenizer.texts_to_sequences(train_phrases)
encoded_test_phrases = tokenizer.texts_to_sequences(test_phrases)

#Watch for a POST padding, as opposed to the default PRE padding

X_train_words = sequence.pad_sequences(encoded_train_phrases, maxlen=max_len,  padding='post')
X_test_words = sequence.pad_sequences(encoded_test_phrases, maxlen=max_len,  padding='post')
print (X_train_words.shape)
print (X_test_words.shape)
print (X_train_target_binary.shape)

print ('Done Tokenizing and indexing phrases based on the vocabulary learned from the entire Train and Test corpus')

word_index = tokenizer.word_index
embeddings_index = {}
embedding_size = 300

with open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec', 'r') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
num_words = min(vocab_size, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_size))
for word, i in word_index.items():
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
print('Done building embedding matrix from FastText')

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import  GlobalMaxPool1D, SpatialDropout1D
from keras.layers import Bidirectional
from keras.models import Model
from keras import optimizers

early_stop = EarlyStopping(monitor = "val_loss", mode="min", patience = 3, verbose=1)

print("Building layers")        
nb_epoch = 15
print('starting to stitch and compile  model')

# Embedding layer for text inputs
input_words = Input((max_len, ))
x_words = Embedding(num_words, embedding_size,weights=[embedding_matrix],trainable=False)(input_words)
x_words = SpatialDropout1D(0.3)(x_words)
x_words = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x_words)
x_words = Dropout(0.2)(x_words)
x_words = Conv1D(128, 1, strides = 1,  padding='causal', activation='relu', )(x_words)
x_words = Conv1D(256, 3, strides = 1,  padding='causal', activation='relu', )(x_words)
x_words = Conv1D(512, 5, strides = 1,   padding='causal', activation='relu', )(x_words)
x_words = GlobalMaxPool1D()(x_words)
x_words = Dropout(0.2)(x_words)

x = Dense(50, activation="relu")(x_words)
x = Dropout(0.2)(x)
predictions = Dense(5, activation="softmax")(x)

test_preds = np.zeros((test_raw.shape[0], 5))

for i in range(NUM_FOLDS):
    print("FOLD", i+1)
    train, val = X_train_words[train_raw["fold_id"] != i], X_train_target_binary[train_raw["fold_id"] != i]
    
    print("Building the model...")
    model = Model(inputs=[input_words], outputs=predictions)
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["acc"])
    
    print("Training the model...")
    history = model.fit([train], 
                        val, 
                        epochs=nb_epoch, 
                        verbose=1, 
                        batch_size = 1024, 
                        callbacks=[early_stop], 
                        validation_split = 0.2, 
                        shuffle=True)
    print("Predicting...")
    test_preds += model.predict([X_test_words], batch_size=1024, verbose = 0)
    
print("Make the submission ready...")
submission = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
submission.Sentiment = np.round(np.argmax(test_preds, axis=1)).astype(int)
submission.to_csv('submission.csv',index=False)
