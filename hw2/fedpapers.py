import numpy as np
import json
import math
from sklearn.feature_extraction import text

x = open('fedpapers_split.txt').read()
papers = json.loads(x)

papersH = papers[0] # papers by Hamilton 
papersM = papers[1] # papers by Madison
papersD = papers[2] # disputed papers

nH, nM, nD = len(papersH), len(papersM), len(papersD)
#for index in range(nD-1):

# This allows you to ignore certain common words in English
# You may want to experiment by choosing the second option or your own
# list of stop words, but be sure to keep 'HAMILTON' and 'MADISON' in
# this list at a minimum, as their names appear in the text of the papers
# and leaving them in could lead to unpredictable results
stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON','MADISON'})
#stop_words = {'HAMILTON','MADISON'}

## Form bag of words model using words used at least 10 times
vectorizer = text.CountVectorizer(stop_words,min_df=10)
X = vectorizer.fit_transform(papersH+papersM+papersD).toarray()

# Uncomment this line to see the full list of words remaining after filtering out 
# stop words and words used less than min_df times
#vectorizer.vocabulary_

# Split word counts into separate matrices
XH, XM, XD = X[:nH,:], X[nH:nH+nM,:], X[nH+nM:,:]

# Estimate probability of each word in vocabulary being used by Hamilton
word_frequency_H = XH.sum(axis=0)
sum_word_frequency_H = sum(word_frequency_H)
fH = []
for index in range(len(word_frequency_H)-1):
	prob = float(word_frequency_H[index]+1)/(sum_word_frequency_H + len(word_frequency_H))
	fH.append(prob)
	print "fH",  prob

# Estimate probability of each word in vocabulary being used by Madison
word_frequency_M = XM.sum(axis=0)
sum_word_frequency_M = sum(word_frequency_M)
fM = []
for index in range(len(word_frequency_M)-1):
	prob = float(word_frequency_M[index]+1)/(sum_word_frequency_M + len(word_frequency_M))
	fM.append(prob)

# Compute ratio of these probabilities
#fratio = fH/fM
fratio = []
for index in range(len(word_frequency_M)-1):
	ratio = fH[index]/fM[index]
	fratio.append(ratio)

# Compute prior probabilities
n = nH + nM 
piH = float(nH)/n
piM = float(nM)/n


count = 0
for xd in XD: # Iterate over disputed documents
    # Compute likelihood ratio for Naive Bayes model
	fX_ratio = 1
	for index in range(len(xd)-1):
		fX_ratio = fX_ratio*math.pow(fratio[index], xd[index])
    	LR = fX_ratio * (piH/piM)
    	if LR>1:
        	print 'Hamilton'
    	else:
        	print 'Madison'
	
    
