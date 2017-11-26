import os
import pandas as pd
import numpy as np
import re
import nltk as nl

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Tensorflow')

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)

lines = df['Review Text']

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def removeStopWords(string):

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(string)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)

    return filtered_sentence


#
# from nltk.corpus import stopwords
#
#
# example_sent = "This is a sample sentence, showing off the stop words filtration."
#
# stop_words = set(stopwords.words('english'))
#
# word_tokens = word_tokenize(example_sent)
#
# filtered_sentence = [w for w in word_tokens if not w in stop_words]
#
# filtered_sentence = []
#
# for w in word_tokens:
#     if w not in stop_words:
#         filtered_sentence.append(w)
#
# print(word_tokens)
# print(filtered_sentence)



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

from nltk import PorterStemmer
stemmer = PorterStemmer()

def tokenise(string):
    string = self.clean(string)
    words = string.split(" ")
    return ([self.stemmer.stem(word,0,len(word)-1) for word in words])

tokenise('what a ridiculous sentence I loved to thing going')

df['Review Text'] = df['Review Text'].apply(cleanstring)

df['Review Text'] = df['Review Text'].apply(removeStopWords)

df
