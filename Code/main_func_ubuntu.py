import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import sys
import csv
import codecs
import io
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
from sklearn.neural_network import MLPClassifier

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import ngrams


from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB



#fr = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train.txt',  'r' , encoding="utf8")
#fw = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\feature1.txt',  'w' , encoding="utf8")
#ftr = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\feature1.txt', 'r', encoding="utf8")

fr = io.open('/home/nirob/Dropbox/ML/DSL-Task-master/data/DSLCC-v2.0/train-dev/train1000.txt', 'r', encoding="utf8")
ftr = io.open('/home/nirob/Dropbox/ML/DSL-Task-master/data/DSLCC-v2.0/train-dev/feature_ubuntu200_cut.txt', 'r', encoding="utf8")
fw = io.open('/home/nirob/Dropbox/ML/DSL-Task-master/data/DSLCC-v2.0/train-dev/feature_ubuntu.txt', 'w', encoding="utf8")

language_list = []
line_list = []
for line in fr:
	line_list.append(line)
	word_list = line.split()
	language_list.append(word_list[-1])


def func_word_gram_freq(algo):

	feature= []
	for line in ftr:
		words = line.split();
		for item in words:
			feature.append(item)

	num_row_word = 2800
	num_col_word = len(feature)

	print(num_col_word)

	if(algo == 2):
		num_col_word += 1

	feature_matrix_word = [[0 for x in range(num_col_word)] for y in range(num_row_word)]
	feature_matrix_word_shuffle = [[0 for x in range(num_col_word)] for y in range(num_row_word)]

	for i in range(0,num_row_word):
		for j in range(0,num_col_word):
			# if(algo == 2 and j == num_col_word - 1):
			# 	feature_matrix_word[i][j] = language_list[i]
			# 	continue

			word_list = line_list[i].split()
			for item in word_list:
				if item==feature[j]:
					feature_matrix_word[i][j] = 1

	if(algo == 1):
		logreg_model_word = LogisticRegression(C=1e5)
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(logreg_model_word, feature_matrix_word, language_list, cv=cv)
		print(scores_word)

	if(algo == 2):
		feature_matrix_word_shuffle = feature_matrix_word
		np.random.shuffle(feature_matrix_word_shuffle)
		train = 1960
		language_list_shuffle = []

		for i in range(0,num_row_word):
			language_list_shuffle.append(feature_matrix_word_shuffle[i][num_col_word - 1])

		for row in feature_matrix_word_shuffle:
			del row[num_col_word - 1]

		trainMatrix = feature_matrix_word_shuffle[:train]
		testMatrix = feature_matrix_word_shuffle[train:]
		trainLabel = language_list_shuffle[:train]
		testLabel = language_list_shuffle[train:]
		
		clf = GaussianNB()
		clf.fit(trainMatrix, trainLabel)
		predicted = clf.predict(testMatrix)
		print(accuracy_score(testLabel,predicted))
		print(predicted)

	if(algo == 3):
		nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 5), random_state=1)
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(nn, feature_matrix_word, language_list, cv=cv)
		print(scores_word)

	if(algo == 4):
		#clf = svm.SVC(kernel='linear')
		#clf = svm.SVC(kernel='rbf')
		clf = svm.SVC(kernel='linear')
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(clf, feature_matrix_word, language_list, cv=cv)
		print(scores_word)



def func_word_gram(gram, feature, algo):
	vectorizer_word = TfidfVectorizer(encoding = "utf8", ngram_range=(gram, gram), max_features = feature)
	X_train_word = vectorizer_word.fit_transform(line_list)
	features_word = vectorizer_word.get_feature_names();


	num_row_word = 14000
	num_col_word = feature

	if(algo == 2):
		num_col_word += 1

	feature_matrix_word = [[0 for x in range(num_col_word)] for y in range(num_row_word)]
	feature_matrix_word_shuffle = [[0 for x in range(num_col_word)] for y in range(num_row_word)]

	count = 0
	for i in range(0,num_row_word):
		for j in range(0,num_col_word):
			if(algo == 2 and j == num_col_word - 1):
				feature_matrix_word[i][j] = language_list[i]
				continue

			word_list = ngrams(line_list[i].split(), gram)
			for grams in word_list:
				if gram==1 and grams[0]==features_word[j]:
					feature_matrix_word[i][j] = 1
				elif gram==2 and (grams[0]+' '+grams[1]) == features_word[j]:
					feature_matrix_word[i][j] = 1
				elif gram==3 and (grams[0]+' '+grams[1]+' '+grams[2]) == features_word[j]:
					feature_matrix_word[i][j] = 1
				elif gram==4 and (grams[0]+' '+grams[1]+' '+grams[2]+' '+grams[3]) == features_word[j]:
					feature_matrix_word[i][j] = 1
				elif gram==5 and (grams[0]+' '+grams[1]+' '+grams[2]+' '+grams[3]+' '+grams[4]) == features_word[j]:
					feature_matrix_word[i][j] = 1
				elif gram==6 and (grams[0]+' '+grams[1]+
					' '+grams[2]+' '+grams[3]+' '+grams[4]+' '+grams[5]) == features_word[j]:
					feature_matrix_word[i][j] = 1
					count+=1
					#print(count)
				elif gram==7 and (grams[0]+' '+grams[1]+' '+grams[2]+' '+grams[3]+' '+grams[4]+' '+grams[5]+' '+grams[6]) == features_word[j]:
					feature_matrix_word[i][j] = 1


	if(algo == 1):
		logreg_model_word = LogisticRegression(C=1e5)
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(logreg_model_word, feature_matrix_word, language_list, cv=cv)
		print(scores_word)

	if(algo == 2):
		feature_matrix_word_shuffle = feature_matrix_word
		np.random.shuffle(feature_matrix_word_shuffle)
		train = 9800
		language_list_shuffle = []
		# trainMatrix = feature_matrix_word_shuffle[:train]
		# print(trainMatrix)
		
		#trainMatrix2 = np.array(feature_matrix_word_shuffle)[:train,:num_col_word-1].tolist()
		#testMatrix2 = np.array(feature_matrix_word_shuffle)[train:,:num_col_word-1].tolist()
		#trainLabel2 = list(np.array(feature_matrix_word_shuffle)[:train,num_col_word-1:])
		# testLabel = np.array(feature_matrix_word_shuffle)[train:,num_col_word-1:].ravel()
		for i in range(0,num_row_word):
			language_list_shuffle.append(feature_matrix_word_shuffle[i][num_col_word - 1])

		for row in feature_matrix_word_shuffle:
			del row[num_col_word - 1]

		trainMatrix = feature_matrix_word_shuffle[:train]
		testMatrix = feature_matrix_word_shuffle[train:]
		trainLabel = language_list_shuffle[:train]
		testLabel = language_list_shuffle[train:]
		
		#print(testMatrix)
		#print(trainLabel)
		#print(testMatrix2)
		# trainLabel2 = language_list[:train]
		# print(trainLabel)
		# print(trainLabel2)
		#testLabel = language_list[train:]
		#print(trainLabel)
		clf = GaussianNB()
		clf.fit(trainMatrix, trainLabel)
		predicted = clf.predict(testMatrix)
		print(accuracy_score(testLabel,predicted))
		print(predicted)

	if(algo == 5):
		logreg_model_word = GaussianNB()
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(logreg_model_word, feature_matrix_word, language_list, cv=cv)
		print(scores_word)

	if(algo == 3):
		nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1)
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(nn, feature_matrix_word, language_list, cv=cv)
		print(scores_word)

	if(algo == 6):
		#clf = svm.SVC(kernel='linear')
		#clf = svm.SVC(kernel='rbf')
		clf = svm.SVC(kernel='linear')
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(clf, feature_matrix_word, language_list, cv=cv)
		print(scores_word)



def func_character_gram(gram, feature, algo):
	vectorizer_char = TfidfVectorizer(encoding = "utf8", analyzer = 'char', ngram_range=(gram, gram), max_features = feature)
	X_train_char = vectorizer_char.fit_transform(line_list)
	features_char = vectorizer_char.get_feature_names();


	num_row_char = 252000
	num_col_char = feature

	if(algo == 2):
		num_col_char += 1

	feature_matrix_char = [[0 for x in range(num_col_char)] for y in range(num_row_char)]
	feature_matrix_char_shuffle = [[0 for x in range(num_col_char)] for y in range(num_row_char)]

	#for grams in features_char:
	#	fw.write(grams+ '\n')
		#print(grams)
	count = 0
	for i in range(0,num_row_char):
		for j in range(0,num_col_char):
			if(algo == 2 and j == num_col_char - 1):
				feature_matrix_char[i][j] = language_list[i]
				continue

			word_list = [line_list[i][i:i+gram] for i in range(len(line_list[i])-1)]
			for grams in word_list:
				if grams==features_char[j]:
					feature_matrix_char[i][j] = 1
				# elif gram==2 and (grams[0]+grams[1]) == features_char[j]:
				# 	feature_matrix_char[i][j] = 1
				# elif gram==3 and (grams[0]+grams[1]+grams[2]) == features_char[j]:
				# 	feature_matrix_char[i][j] = 1
				# elif gram==4 and (grams[0]+grams[1]+grams[2]+grams[3]) == features_char[j]:
				# 	feature_matrix_char[i][j] = 1
				# elif gram==5 and (grams[0]+grams[1]+grams[2]+grams[3]+grams[4]) == features_char[j]:
				# 	feature_matrix_char[i][j] = 1
				# elif gram==6 and (grams[0]+grams[1]+grams[2]+grams[3]+grams[4]+grams[5]) == features_char[j]:
				# 	feature_matrix_char[i][j] = 1
				# 	count+=1
				# 	#print(count)
				# elif gram==7 and (grams[0]+grams[1]+grams[2]+grams[3]+grams[4]+grams[5]+grams[6]) == features_char[j]:
				# 	feature_matrix_char[i][j] = 1


	if(algo == 1):
		logreg_model_word = LogisticRegression(C=1e5)
		cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores_word = cross_val_score(logreg_model_word, feature_matrix_char, language_list, cv=cv)
		print(scores_word)

	if(algo == 2):
		feature_matrix_char_shuffle = feature_matrix_char
		np.random.shuffle(feature_matrix_char_shuffle)
		train = 1960
		language_list_shuffle = []
		# trainMatrix = feature_matrix_char_shuffle[:train]
		# print(trainMatrix)
		
		#trainMatrix2 = np.array(feature_matrix_char_shuffle)[:train,:num_col_char-1].tolist()
		#testMatrix2 = np.array(feature_matrix_char_shuffle)[train:,:num_col_char-1].tolist()
		#trainLabel2 = list(np.array(feature_matrix_char_shuffle)[:train,num_col_char-1:])
		# testLabel = np.array(feature_matrix_char_shuffle)[train:,num_col_char-1:].ravel()
		for i in range(0,num_row_char):
			language_list_shuffle.append(feature_matrix_char_shuffle[i][num_col_char - 1])

		for row in feature_matrix_char_shuffle:
			del row[num_col_char - 1]

		trainMatrix = feature_matrix_char_shuffle[:train]
		testMatrix = feature_matrix_char_shuffle[train:]
		trainLabel = language_list_shuffle[:train]
		testLabel = language_list_shuffle[train:]
		
		# print(testMatrix)
		# print(trainLabel)
		# print(testMatrix2)
		# trainLabel2 = language_list[:train]
		# print(trainLabel)
		# print(trainLabel2)
		#testLabel = language_list[train:]
		#print(trainLabel)
		clf = GaussianNB()
		clf.fit(trainMatrix, trainLabel)
		predicted = clf.predict(testMatrix)
		print(accuracy_score(testLabel,predicted))
		#print(predicted)


func_word_gram(1,200,6)
#func_word_gram_freq(4)
#func_character_gram(1,200,1)