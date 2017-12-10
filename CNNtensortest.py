import os

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Data')

import pandas as pd

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)

length = []
import nltk as nl
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))

lines = df['Review Text']

import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import tensorflow.contrib.learn as tflearn

#model parameters

BATCH_SIZE = 32
EMBEDDING_SIZE = 10
WINDOW_SIZE = EMBEDDING_SIZE
STRIDE = int(WINDOW_SIZE/2)

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
vocab_processor.fit(lines)
# with gfile.Open('vocab.tsv', 'wb') as f:
#     f.write("{}\n".format(PADWORD))
#     for word, index in vocab_processor.vocabulary_._mapping.items():
#       f.write("{}\n".format(word))
N_WORDS = len(vocab_processor.vocabulary_)

def train_input_fn():

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    filename = "train.csv"
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = DEFAULTS
    col1, col2= tf.decode_csv(
        value, record_defaults=record_defaults, field_delim='|')
    label = tf.stack([col1])
    features = tf.stack([col2])
    # features = dict(zip('Review Text', col2))

    # For multiclass classification use longer 'TARGETS' attribute
    table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    labels = table.lookup(label)

    return features, labels

train_input_fn()

def eval_input_fn():

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    filename = "eval.csv"
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = DEFAULTS
    col1, col2= tf.decode_csv(
        value, record_defaults=record_defaults, field_delim='|')
    label = tf.stack([col1])
    features = tf.stack([col2])
    # features = dict(zip('Review Text', col2))

    # For multiclass classification use longer 'TARGETS' attribute
    table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    labels = table.lookup(label)

    return features, labels

#pandas can't be used because it wants feature columns, ours needs processing inside the model

def cnn_model(features, labels, mode):

    # convert vocab to numbers
    table = lookup.index_table_from_file(
      vocabulary_file='vocab.tsv', num_oov_buckets=1, vocab_size=None, default_value=-1)

    #Looks up specific terms 'Some title'
    # numbers = table.lookup(tf.constant('Some title'.split()))
    # with tf.Session() as sess:
    #   tf.tables_initializer().run()
    #   print("{} --> {}".format(lines[0], numbers.eval()))

    #create sparse vectors, convert to dense and look vectors up in the dictionary
    # titles = tf.squeeze(features['Review Text'], [1])
    words = tf.string_split(features)
    densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table.lookup(densewords)

    #Shows dense word vectors
    # sess = tf.Session()
    #sess.run(densewords)

    #Shows vectors of words where dictionary is applied
    #table.init.run(session=sess)
    #print(numbers.eval(session=sess))

    #pads vectors out to MAX_DOCUMENT_LENGTH
    padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])
    # sess.run(sliced)

    #create embeddings

    embeds = tf.contrib.layers.embed_sequence(sliced, vocab_size=N_WORDS, embed_dim=EMBEDDING_SIZE)
    #print('words_embed={}'.format(embeds)) # (?, 20, 10)

    #Convolutions!!!

    conv = tf.contrib.layers.conv2d(embeds, 1, WINDOW_SIZE,
                    stride=STRIDE, padding='SAME') # (?, 4, 1)
    conv = tf.nn.relu(conv) # (?, 4, 1)
    words = tf.squeeze(conv, [2]) # (?, 4)

    logits = tf.contrib.layers.fully_connected(words, n_classes, activation_fn=None)

    predictions_dict = {
      'Fiction': tf.gather(TARGETS, tf.argmax(logits, 1)),
      'prob': tf.nn.softmax(logits)
    }

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
       loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
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

output_dir = '/Users/clementmanger/Desktop/Thesis/Tensorflow/TFCNN'

cnn = tf.estimator.Estimator(model_fn=cnn_model, config=tflearn.RunConfig(model_dir=output_dir))

cnn.train(input_fn=train_input_fn, steps = 300)

# ev = cnn.evaluate(input_fn=eval_input_fn, steps = 5)
#
# pred = cnn.predict(input_fn=eval_input_fn)
#
# for i, p in enumerate(pred):
#
#     print("Prediction %s: %s" % (i + 1, p["prob"]))


# print('hooray')

#try running it outside of hydrogen

# #sort it out and put it in an experiment wrapper
# from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
#
# TRAIN_STEPS = 2000
#
# import tensorflow.contrib.metrics as metrics
#
# import tensorflow.contrib.learn as tflearn
#
# def serving_input_fn():
#     feature_placeholders = {
#       'Review Text': tf.placeholder(tf.string, [None]),
#     }
#     features = {
#       key: tf.expand_dims(tensor, -1)
#       for key, tensor in feature_placeholders.items()
#     }
#     return tflearn.utils.input_fn_utils.InputFnOps(
#       features,
#       None,
#       feature_placeholders)
#
# def experiment_fn(output_dir):
#     # run experiment
#     config = tflearn.RunConfig(model_dir=output_dir)
#     return tf.contrib.learn.Experiment(
#         tflearn.Estimator(model_fn=cnn_model, config=config),
#         train_input_fn=train_input_fn(),
#         eval_input_fn=eval_input_fn(),
#         eval_metrics={
#             'acc': tflearn.MetricSpec(
#                 metric_fn=metrics.streaming_accuracy, prediction_key='Fiction'
#             )
#         },
#         # export_strategies=[saved_model_export_utils.make_export_strategy(
#         #     serving_input_fn,
#         #     default_output_alternative_key=None,
#         #     exports_to_keep=1
#         # )],
#         train_steps = TRAIN_STEPS
#     )
#
# learn_runner.run(experiment_fn, output_dir)
