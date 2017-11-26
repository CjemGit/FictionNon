import os

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis')
cwd

import pandas as pd

df = pd.read_csv('revsample.csv', sep = '|')
df.columns
list1 = df['Review Text'].tolist()

len(list1)

import re

list2 = []

list3 = []

for f in range(len(list1)):

    list2 = re.sub(r'\.([^ ])', r'. \1', list1[f])

    list3.append(list2)

list3

import nltk

test = pd.DataFrame(columns=['Review ID'])

for f in range(len(list3)):

    item = nltk.sent_tokenize(list3[f])

    test2 = pd.DataFrame(item, columns=['Sentence Text'])

    test2['Review ID'] = f

    test = test.append(test2)

test.to_csv('RevSentences.csv')
