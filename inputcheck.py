import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.rnn as rnn

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Data')

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)

df['Fiction']

#figure out document lengths
length = []
import nltk as nl
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))
lines = df['Review Text']

#model parameters
BATCH_SIZE = 1
EMBEDDING_SIZE = 10
LSTM_SIZE = 3

#import parameters
TARGETS = ['True', 'False']
DEFAULTS = [['null'], ['null']]
n_classes = len(TARGETS)
MAX_DOCUMENT_LENGTH = max(length)
PADWORD = 'ZYXW'
FEATURE = 'Review Text'
LABEL = 'Fiction'

# create vocabulary
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
# vocab_processor.fit(lines)
# with gfile.Open('vocab.tsv', 'wb') as f:
#     f.write("{}\n".format(PADWORD))
#     for word, index in vocab_processor.vocabulary_._mapping.items():
#       f.write("{}\n".format(word))

x = np.array(list(vocab_processor.fit_transform(lines)))

vocabulary = vocab_processor.vocabulary_

vocab_size = len(vocabulary)

def train_input_fn():

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    filename = "train.csv"
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    record_defaults = DEFAULTS
    col1, col2= tf.decode_csv(
        value, record_defaults=record_defaults, field_delim='|')
    label = tf.stack([col1])
    features = tf.stack([col2])

    table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    labels = table.lookup(label)

    table2 = lookup.index_table_from_file(
      vocabulary_file='vocab.tsv', num_oov_buckets=1, vocab_size=None, default_value=-1)

    #look strings up in the vocabulary
    words = tf.string_split(features)
    densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table2.lookup(densewords)

    #pads vectors out to MAX_DOCUMENT_LENGTH
    padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])
    shaped = tf.reshape(sliced, [1735])

    batch_size = file_len(filename)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    features, labels = tf.train.shuffle_batch(
      [shaped, labels], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

    return features, labels

with tf.Session() as sess:

    sess.run(train_input_fn()[0])
