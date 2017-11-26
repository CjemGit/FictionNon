import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.rnn as rnn

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Tensorflow')

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)

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

#
# def eval_input_fn():
#
#     def file_len(fname):
#         with open(fname) as f:
#             for i, l in enumerate(f):
#                 pass
#         return i + 1
#
#     filename = "eval.csv"
#     filename_queue = tf.train.string_input_producer([filename])
#     reader = tf.TextLineReader()
#     _, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
#
#     # Default values, in case of empty columns. Also specifies the type of the
#     # decoded result.
#     record_defaults = DEFAULTS
#     col1, col2= tf.decode_csv(
#         value, record_defaults=record_defaults, field_delim='|')
#     label = tf.stack([col1])
#     features = tf.stack([col2])
#     # features = dict(zip('Review Text', col2))
#
#     # For multiclass classification use longer 'TARGETS' attribute
#     table = tf.contrib.lookup.index_table_from_tensor(
#                     mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
#     labels = table.lookup(label)
#
#     return features, labels

#pandas can't be used because it wants feature columns, ours needs processing inside the model


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors



def RNN_model(features, labels, mode):

    initW = load_embedding_vectors_word2vec(vocabulary, '/Users/clementmanger/Desktop/Thesis/Word2Vec/GoogleNews-vectors-negative300.bin', True)

    V = tf.Variable(tf.random_uniform([vocab_size, 300], -1.0, 1.0),name="W")

    V.assign(initW)

    features = tf.nn.embedding_lookup(V, features)

    # features = tf.expand_dims((features), -1)

    #features = tf.bitcast(features, tf.float64)

    labels = tf.squeeze(labels)

    seqlen = tf.placeholder(tf.int32, [features.shape[0]])

    lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, features, dtype=tf.float32)

    #something about this stops linear from receiving 2d arguments
    #, sequence_length=seqlen, initial_state=[features.shape[0], LSTM_SIZE])

    #static RNN requires inputs as a sequence

    #slice to keep only the last cell of the RNN
    outputs = outputs[:, -1]

    #softmax layer

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [LSTM_SIZE, n_classes], dtype=tf.float32)
        b = tf.get_variable('b', [n_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    logits = tf.matmul(outputs, W) + b

    predictions_dict = {
      'Fiction': tf.gather(TARGETS, tf.argmax(logits, 1)),
      'prob': tf.nn.softmax(logits)
    }

    #for softmax cross entropy, logits and labels must have same dimensionality, this means the logits must be the same as batch size

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
       loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
       train_op = tf.contrib.layers.optimize_loss(
         loss,
         tf.contrib.framework.get_global_step(),
         optimizer='Adam',
         learning_rate=0.01)
    else:
       loss = None
       train_op = None

    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions=predictions_dict)

output_dir = '/Users/clementmanger/Desktop/Thesis/Tensorflow/TFGw2v'

RNN = tf.estimator.Estimator(model_fn=RNN_model, config=tflearn.RunConfig(model_dir=output_dir))

RNN.train(input_fn=train_input_fn, steps = 98)

# ev = RNN.evaluate(input_fn=eval_input_fn, steps = 100)
