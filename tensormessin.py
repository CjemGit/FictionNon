import os

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Data')

import pandas as pd

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)

import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile

#
# #figure out document lengths
length = []
import nltk as nl
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))
#
# lines = df['Review Text']
#
#
# BATCH_SIZE = 20
# EMBEDDING_SIZE = 10
# LSTM_SIZE = 3
#
# #import parameters
TARGETS = ['True', 'False']
DEFAULTS = [['null'], ['null']]
# n_classes = len(TARGETS)
MAX_DOCUMENT_LENGTH = max(length)
PADWORD = 'ZYXW'
# FEATURE = 'Review Text'
# LABEL = 'Fiction'
#
# # create vocabulary
# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
# vocab_processor.fit(lines)
#
# N_WORDS = len(vocab_processor.vocabulary_)

import numpy
numpy.set_printoptions(threshold=40)
#

#
# #figure out how much data is being read in
#
import tensorflow as tf
import numpy as np
from tensorflow.contrib import lookup
tf.reset_default_graph()

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
# shaped = tf.expand_dims(shaped, -1)

batch_size = file_len(filename)
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size

example_batch, label_batch = tf.train.shuffle_batch(
  [shaped, labels], batch_size=batch_size, capacity=capacity,
  min_after_dequeue=min_after_dequeue)

import numpy
numpy.set_printoptions(threshold=numpy.nan)

with tf.Session() as sess:
    # Start populating the filename queue.
    tf.tables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(batch_size):

    # print(sess.run(tf.squeeze(label_batch)))
    # print(tf.expand_dims((example_batch), -1).shape)
    # print(tf.squeeze(labels).shape)
    # print(sess.run(tf.expand_dims((example_batch), -1)))
    # print(sess.run((example_batch)))
    # print(tf.squeeze(labels).shape)

    #THIS READS FROM THE SAME
        print(sess.run([shaped]))

    #THESE READ FROM DIFFERENT ONES
    # print(densewords.shape)
    # print(numbers.shape)
    # print(padded.shape)
    # print(sliced.shape)
    # print(shaped.shape)

    coord.request_stop()
    coord.join(threads)

#the way tensorflow sends information around is such that print operations may refer to different data streams. Work out a way of reading the same stream


# import tensorflow as tf
# import numpy as np
# tf.reset_default_graph()
#
# BATCH = 1
# num_examples = 5
# num_features = 3
# data = np.reshape(np.arange(num_examples*num_features), (num_examples, num_features))
#
# (data_node,) = tf.train.slice_input_producer([tf.constant(data)], num_epochs=1, shuffle=False)
# data_node_debug = tf.Print(data_node, [data_node], "Dequeueing from data_node ")
# data_batch = tf.train.batch([data_node_debug], batch_size=BATCH)
# data_batch_debug = tf.Print(data_batch, [data_batch], "Dequeueing from data_batch ")
#
# with tf.Session() as sess:
#     # Start populating the filename queue.
#     sess.run(tf.local_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#       while True:
#         print(sess.run(data_batch_debug))
#     except tf.errors.OutOfRangeError as e:
#       print("No more inputs.")
#
#     coord.request_stop()
#     coord.join(threads)
