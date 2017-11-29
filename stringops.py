import os
import pandas as pd
import numpy as np
import re
import nltk as nl

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Data')

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

from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize(string):
    lmtzr = WordNetLemmatizer()
    word_tokens = word_tokenize(string)

    lemmaz = []
    for word in word_tokens:
        lemmaz.append(lmtzr.lemmatize(word))
    lemmaz = ' '.join(lemmaz)

    return lemmaz


def Propernoun(string):
    word_tokens = word_tokenize(string)
    sent = []
    for x in nl.pos_tag(word_tokens):
        if x[1] == 'NNP':
            sent.append(x[1])
        else:
            sent.append(x[0])
    sent = ' '.join(sent)
    return sent

df['Review Text'] = df['Review Text'].apply(Propernoun)

df['Review Text'] = df['Review Text'].apply(cleanstring)

df['Review Text'] = df['Review Text'].apply(removeStopWords)

df['Review Text'] = df['Review Text'].apply(lemmatize)

df
