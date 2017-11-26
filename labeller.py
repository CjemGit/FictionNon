import os
cwd = os.getcwd()

cwd
import pandas as pd
import numpy as np

number = input("How many are you going to label?")

revs = pd.read_csv('clemrevs.csv', delimiter='|', index_col=0)

revs

labelled = revs[revs['fiction'] == np.NaN]

revs = revs[revs['fiction'] != np.NaN]

sample = revs.sample(int(number))

revs = pd.concat([revs, sample]).drop_duplicates(keep=False)

lsample = pd.DataFrame(columns=["Review Text", 'fiction'])

lrow = pd.DataFrame(columns=["Review Text", 'fiction'])

for row in sample.iterrows():

	print(" ")
	print(row[1]['Review Text'])

	while True:

		var = input("Fiction or Non Fiction? Enter f/n")

		if var == "n":

			lrow.loc[-1] = [row[1]['Review Text'], False]

			break

		elif var == "f":

			lrow.loc[-1] = [row[1]['Review Text'], True]

			break
				# create a new row with this review text and fiction == true
		else:
			print("you have made an invalid choice, try again.")

	lsample = pd.concat([lsample, lrow])

labelled = pd.concat([labelled, lsample])

revs = pd.concat([revs, labelled])

revs.loc[revs['fiction'] == np.nan]
print("")
print("You have labelled " + str(len(revs[revs['fiction'].notnull()])))
print("")
print(revs[revs['fiction'].notnull()])
print("")
print("You still need to label " + str(len(revs[revs['fiction'].isnull()])))

revs.to_csv('clemrevs.csv', sep='|')

os.chdir('/Users/clementmanger/Documents')

revs.to_csv('backup.csv', sep='|')
