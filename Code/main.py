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
from sklearn.model_selection import ShuffleSplit


import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

unigram_set = set()
unigram_list = []
unigram_map = {}



fr = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train200.txt',  'r' , encoding="utf8")
fw = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\feature1.txt',  'w' , encoding="utf8")


language_list = []
line_list = []
for line in fr:
	line_list.append(line)
	word_list = line.split()
	language_list.append(word_list[-1])


vectorizer_word_1 = TfidfVectorizer(encoding = "utf8", max_features = 200)
X_train_word_1 = vectorizer_word_1.fit_transform(line_list)
features_word_1 = vectorizer_word_1.get_feature_names();

vectorizer_word_2 = TfidfVectorizer(encoding = "utf8", ngram_range=(2, 2), max_features = 200)
X_train_word_2 = vectorizer_word_2.fit_transform(line_list)
features_word_2 = vectorizer_word_2.get_feature_names();

#for item in features_word_2:
	#print(item.encode("utf8"))
	#fw.write(item + '\n')

num_row_word_1 = 2800
num_col_word_1 = 200

num_row_word_2 = 2800
num_col_word_2 = 200


feature_matrix_word_1 = [[0 for x in range(num_col_word_1)] for y in range(num_row_word_1)]
feature_matrix_word_2 = [[0 for x in range(num_col_word_2)] for y in range(num_row_word_2)]


for i in range(0,num_row_word_1):
	for j in range(0,num_col_word_1):
		word_list = line_list[i].split()
		for item in word_list:
			if item==features_word_1[j]:
				feature_matrix_word_1[i][j] = 1


from nltk import ngrams

for i in range(0,num_row_word_2):
	for j in range(0,num_col_word_2):
		word_list = ngrams(line_list[i].split(), 2)
		#print(word_list)
		for grams in word_list:
			if ((grams[0]+' '+grams[1]) == features_word_2[j]):
				feature_matrix_word_2[i][j] = 1
			#print(item.encode("utf-8"))
			#fw.write(grams[0] + ' ' + grams[1])



from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

logreg_model_word_1 = LogisticRegression(C=1e5)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores_word_1 = cross_val_score(logreg_model_word_1, feature_matrix_word_1, language_list, cv=cv)
print(scores_word_1)


logreg_model_word_2 = LogisticRegression(C=1e5)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores_word_2 = cross_val_score(logreg_model_word_2, feature_matrix_word_2, language_list, cv=cv)
print(scores_word_2)