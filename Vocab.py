import pandas as pd

df = pd.DataFrame.from_csv('ReviewsFiction.csv', sep = '|', header=0)

#figure out document lengths
length = []
import nltk as nl
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))

lines = df['Review Text']

MAX_DOCUMENT_LENGTH = max(length)
PADWORD = 'ZYXW'

# create vocabulary
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
vocab_processor.fit(lines)
with gfile.Open('vocab.tsv', 'wb') as f:
    f.write("{}\n".format(PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
      f.write("{}\n".format(word))
