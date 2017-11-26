import pandas as pd
import numpy as np
import os

os.getcwd()

df = pd.DataFrame.from_csv('Rev.tsv', sep='\t', header=0)

df = df[pd.notnull(df['Fiction'])]

df = df.reset_index(drop=False)

df.columns

df = df[['Sentence Text', 'Review ID', 'Fiction']]

df2 = pd.DataFrame(columns=['Review Text', 'Fiction'])

row1 = df.iloc[0]

Review = row1['Sentence Text']

for index, row in df.iterrows():

    if row['Review ID'] == row1['Review ID']:

        Review = Review + ' ' + row['Sentence Text']

    else:

        d = {'Review Text': Review, 'Fiction': row['Fiction']}

        df2 = df2.append(pd.DataFrame(data=d, columns=['Review Text', 'Fiction'], index=[0]))

        row1 = row

        Review = row1['Sentence Text']


df2.to_csv('ReviewsFiction.csv', sep = '|')
