import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import sys
import csv
import codecs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
import pickle
from sklearn.externals import joblib
from sklearn.datasets import load_files
import math
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score



import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#fr = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train.txt',  'r' , encoding="utf8")
#fw = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train2.txt',  'w' , encoding="utf8")

unigram_set = set()
unigram_list = []
unigram_map = {}
language_list = []

"""

for line in fr:
	word_list = line.split()
	
	#for item in word_list:
	#	unigram_set.add(item)
	
	#for item in word_list:
	#	unigram_list.append(item)

	language_list.append(word_list[-1])

	unigram_add = unigram_set.add
	[x for x in word_list if not (x in unigram_set or unigram_add(x))]
	
	#[unigram_list.append(item) for item in word_list if item not in unigram_list]
	#print(len(unigram_list))

count = 0
for item in unigram_set:
	unigram_list.append(item)
	unigram_map[item] = count
	#fw.write(str(unigram_map[item]) + " " + item + '\n')
	count += 1



print(len(unigram_list))
print(len(language_list))

"""

line_list = []

fr = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train200.txt',  'r' , encoding="utf8")
fw = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\feature1.txt',  'w' , encoding="utf8")


for line in fr:
	line_list.append(line)
	word_list = line.split()
	language_list.append(word_list[-1])

# for l in line_list:
# 	print(l.encode("utf-8"))
# 	break

vectorizer = TfidfVectorizer(encoding = "utf8", max_features=5000)
X_train = vectorizer.fit_transform(line_list)

#print(X_train.shape)

features_word_1 = vectorizer.get_feature_names();

feature_count = 0

for item in features_word_1:
	unigram_map[item] = feature_count
	feature_count += 1
	fw.write(item + '\n')
	#print(item.encoding="utf8")



num_row_word_1 = 2800
num_col_word_1 = 5000


feature_matrix_word_1 = [[0 for x in range(num_col_word_1)] for y in range(num_row_word_1)]


for i in range(0,num_row_word_1):
	for j in range(0,num_col_word_1):
		word_list = line_list[i].split()
		for item in word_list:
			if item==features_word_1[j]:
				feature_matrix_word_1[i][j] = 1



# line_count = 0

# for line in fr:
# 	word_list = line.split()
# 	for item in word_list:
# 		feature_matrix_word_1[line_count][unigram_map[item]] = 1 
# 	line_count += 1


#dump(feature_matrix_word_1,"D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\feature_matrix_1.dat")

#with open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\feature_matrix_1.dat','wb') as f:
 #   pickle.dump(feature_matrix_word_1,f)

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression(C=1e5)


scores = cross_val_score(logreg_model, feature_matrix_word_1, language_list, cv=5)
print(scores)



# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(feature_matrix_word_1,language_list)
# predicted = clf.predict(testMatrix)
# print(accuracy_score(testLabel,predicted))