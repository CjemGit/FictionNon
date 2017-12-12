import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.rnn as rnn
import datetime

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Data')

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)

import re

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
    return string.strip().lower()


df['Review Text'] = df['Review Text'].apply(cleanstring)

#writes each review as seperate .txt file
num = 1

for row in df.iterrows():
    if row[1][1] == True:
        os.chdir('/Users/clementmanger/Desktop/Thesis/Data/Fiction')
        with open(str(num) + "fiction.txt", "w", encoding='utf8') as text_file:
            text_file.write(row[1][0])
    else:
        os.chdir('/Users/clementmanger/Desktop/Thesis/Data/NonFiction')
        with open(str(num) + "nonfiction.txt", "w", encoding='utf8') as text_file:
            text_file.write(row[1][0])
    num = num + 1

#makes the vocab a list
import csv
os.chdir('/Users/clementmanger/Desktop/Thesis/Data')

blob = []
with open("vocab.tsv") as tsv:
    for line in csv.reader(tsv, delimiter="\t"):
        blob.append(line)

vocab = []
for i in blob: vocab.append(i[0])

#load vectors

# firstSentence = np.zeros((8), dtype='int32')
# firstSentence[0] = vocab.index("As")
# firstSentence[1] = vocab.index("with")
# firstSentence[2] = vocab.index("most")
# firstSentence[3] = vocab.index("of")
# firstSentence[4] = vocab.index("her")
# firstSentence[5] = vocab.index("books")

length = []
import nltk as nl
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))
lines = df['Review Text']

import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile

#
# #figure out document lengths
length = []
import nltk as nl
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))
MAX_DOCUMENT_LENGTH = max(length)
PADWORD = 'ZYXW'

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
vocab_processor.fit(lines)
with gfile.Open('vocab.tsv', 'wb') as f:
    f.write("{}\n".format(PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
      f.write("{}\n".format(word))

x = np.array(list(vocab_processor.fit_transform(lines)))

vocabulary = vocab_processor.vocabulary_

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

# wordVectors = load_embedding_vectors_word2vec(vocabulary, '/Users/clementmanger/Desktop/Thesis/Data/GoogleNews-vectors-negative300.bin', True)

wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

# #check vector load
# with tf.Session() as sess:
#     print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)

#begin importing reviews

from os import listdir
from os.path import isfile, join
FictionFiles = ['Fiction/' + f for f in listdir('Fiction/') if isfile(join('Fiction/', f))]
NonFictionFiles = ['NonFiction/' + f for f in listdir('NonFiction/') if isfile(join('NonFiction/', f))]
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


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
for ff in FictionFiles:
   with open(pf, "r") as f:
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
np.set_printoptions(threshold=np.NaN)

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,40)
            labels.append([1,0])
        else:
            num = randint(41,97)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
numDimensions = 300

import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
# data = tf.cast(data, tf.float64)

data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), dtype=tf.float32)
# weight = tf.cast(weight, tf.float64)
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
# bias = tf.cast(bias, tf.float64)
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       save_path = saver.save(sess, "'/Users/clementmanger/Desktop/Thesis/Example'", global_step=i)
       print("saved to %s" % save_path)
writer.close()
#suspect the problem is that the vocab starts at 1 rather than 0, problem is with the vocabulary
