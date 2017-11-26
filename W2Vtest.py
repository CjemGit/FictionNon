import os
import tensorflow as tf
import pandas as pd
import nltk as nl
import numpy as np

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Word2Vec')
# cwd

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)
lines = df['Review Text']

#figure out document lengths
length = []
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))
MAX_DOCUMENT_LENGTH = max(length)

# create vocabulary

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

x = np.array(list(vocab_processor.fit_transform(lines)))

vocabulary = vocab_processor.vocabulary_

vocab_size = len(vocabulary)

#check if the vocabulary actually has any words in it. it feels like vocab may have the length, but no actual words in it

#this is the idea, and it comes from https://github.com/cahya-wirawan/cnn-text-classification-tf/blob/master/data_helpers.py
# with open('GoogleNews-vectors-negative300.bin', 'rb') as f:
#     header = f.readline()
#     vocab_size, vector_size = map(int, header.split())
#     print(vocab_size, vector_size)


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

initW = load_embedding_vectors_word2vec(vocabulary, 'GoogleNews-vectors-negative300.bin', True)

W = tf.Variable(tf.random_uniform([vocab_size, 300], -1.0, 1.0),name="W")

W.assign(initW)

tf.nn.embedding_lookup(W, features)

#
#
# def load_embedding_vectors_glove(vocabulary, filename, vector_size):
#     # load embedding_vectors from the glove
#     # initial matrix with random uniform
#     embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
#     f = open(filename)
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], dtype="float32")
#         idx = vocabulary.get(word)
#         if idx != 0:
#             embedding_vectors[idx] = vector
#     f.close()
#     return embedding_vectors
