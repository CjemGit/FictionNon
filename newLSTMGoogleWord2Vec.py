import os
SPEC = 'LSTMGoogleW2Vec'
os.chdir('C:\\Users\\Clembo\\Desktop\\Thesis\\Data')
cwd = os.getcwd()

import pandas as pd
import numpy as np
import tensorflow as tf
import nltk as nl

from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.rnn as rnn
import datetime
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import re
import csv

df = pd.read_csv('ReviewsFiction.csv', sep = '|', header=0, index_col=0)

def cleanstring(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub('[^\w ]', '', string)
    return string.strip().lower()

df['Review Text'] = df['Review Text'].apply(cleanstring)

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

#writes each review as seperate .txt file

sample = round(len(df)/10)

# num = 1
#
# for row in df.iterrows():
#     if row[1][1] == True:
#         os.chdir('C:\\Users\\Clembo\\Desktop\\Thesis\\Data\\Fiction')
#         with open(str(num) + "fiction.txt", "w", encoding='utf8') as text_file:
#             text_file.write(row[1][0])
#     else:
#         os.chdir('C:\\Users\\Clembo\\Desktop\\Thesis\\Data\\NonFiction')
#         with open(str(num) + "nonfiction.txt", "w", encoding='utf8') as text_file:
#             text_file.write(row[1][0])
#     num = num + 1

length = []
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))
lines = df['Review Text']

# #figure out document lengths

MAX_DOCUMENT_LENGTH = max(length)
# PADWORD = 'ZYXW'

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
vocab_processor.fit(lines)

os.chdir('C:\\Users\\Clembo\\Desktop\\Thesis\\Data')

# # write vocab to directory
# with gfile.Open('vocab.tsv', 'wb') as f:
#     # f.write("{}\n".format(PADWORD))
#     for word, index in vocab_processor.vocabulary_._mapping.items():
#       f.write("{}\n".format(word))

x = np.array(list(vocab_processor.fit_transform(lines)))

vocabulary = vocab_processor.vocabulary_

# CHECK FOR VOCAB MATCH
## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary2 = list(list(zip(*sorted_vocab))[0])
#

#makes the vocab a list

blob = []
with open("vocab.tsv") as tsv:
    for line in csv.reader(tsv, delimiter="\t"):
        blob.append(line)

vocab = []
for i in blob: vocab.append(i[0])

badguys = list(set(vocab) - set(vocabulary2))

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


# print ('Loaded the Wikipedia word vectors!')

# #check vector load
# with tf.Session() as sess:
#     print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)
#     print(tf.nn.embedding_lookup(wordVectors2,firstSentence).eval().shape)

#begin importing reviews

from os import listdir
from os.path import isfile, join
FictionFiles = ['Fiction/' + f for f in listdir('Fiction/') if isfile(join('Fiction/', f))]
# FictionFiles.remove('Fiction/.DS_Store')
NonFictionFiles = ['NonFiction/' + f for f in listdir('NonFiction/') if isfile(join('NonFiction/', f))]
# NonFictionFiles.remove('NonFiction/.DS_Store')

numWords = []

for pf in FictionFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Fiction files finished')

for nf in NonFictionFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('NonFiction files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters

maxSeqLength = 300

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
for ff in FictionFiles:
   with open(ff, "r") as f:
       indexCounter = 0
       line=f.readline()
       cleanedLine = cleanSentences(line)
       split = cleanedLine.split()
       for word in split:
           try:
               ids[fileCounter][indexCounter] = vocab.index(word)
           except ValueError:
               ids[fileCounter][indexCounter] = 2 #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1

fictioncounter = fileCounter
print('There are ' + str(fileCounter) + ' fiction files')

for nf in NonFictionFiles:
   with open(nf, "r") as f:
       indexCounter = 0
       line=f.readline()
       cleanedLine = cleanSentences(line)
       split = cleanedLine.split()
       for word in split:
           try:
               ids[fileCounter][indexCounter] = vocab.index(word)
           except ValueError:
               ids[fileCounter][indexCounter] = 2 #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1
print('There are '+ str(fileCounter - fictioncounter) + ' Nonfiction files')

#Pass into embedding function and see if it evaluates.

np.save('FidsMatrix', ids)
ids = np.load('FidsMatrix.npy')

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,fictioncounter-(round(sample/2)))
            labels.append([1,0])
        else:
            num = randint(fictioncounter+(round(sample/2)),fileCounter)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

# def getTestBatch():
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         num = randint((fictioncounter-(sample/2)+1),(fictioncounter+(sample/2)-1))
#         if (num <= fictioncounter):
#             labels.append([1,0])
#         else:
#             labels.append([0,1])
#         arr[i] = ids[num-1:num]
#     return arr, labels

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 3001
numDimensions = 300

import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
# data = tf.cast(data, tf.float64)
import numpy as np
wordVectors = load_embedding_vectors_word2vec(vocabulary, 'C:\\Users\\Clembo\\Desktop\\Thesis\\Data\\GoogleNews-vectors-negative300.bin', True)

data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float64)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), dtype=tf.float32)
weight = tf.cast(weight, tf.float64)
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
bias = tf.cast(bias, tf.float64)
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float64))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + SPEC + "/"
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

import time

timer = pd.DataFrame(columns=['Steps', 'Time'])

t0 = time.time()
for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

       t1 = time.time()
       count = pd.DataFrame(columns=['Steps', 'Time'], data=[[i,(t1-t0)]])
       timer = timer.append(count)

   #Save the network every 1000 training iterations
   if (i % 500 == 0 and i != 0):
       save_path = saver.save(sess, "models/" + SPEC + "/ckpt", global_step=i)
       print("saved to %s" % save_path)
writer.close()

#cd C:\Users\Clembo\Desktop\Thesis\Data
#tensorboard --logdir=tensorboard

#http://localhost:6006/
timer.to_csv('timer.csv')
