import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import sys
import csv
import codecs
import io

import pickle
from sklearn.externals import joblib
from sklearn.datasets import load_files
import math


import numpy as np


from nltk import ngrams


from sklearn import metrics



freq = dict()

#fr = io.open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train.txt',  'r' , encoding="utf8")
#fw = io.open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\feature1.txt',  'w' , encoding="utf8")
fr = io.open('/home/nirob/Dropbox/ML/DSL-Task-master/data/DSLCC-v2.0/train-dev/train200.txt', 'r', encoding="utf8")
fw = io.open('/home/nirob/Dropbox/ML/DSL-Task-master/data/DSLCC-v2.0/train-dev/feature_ubuntu200.txt', 'w', encoding="utf8")

language_list = []
line_list = []
for line in fr:
	word_list = line.split()
	for item in word_list:
		if(item != word_list[-1]):
			if(item in freq):
				freq[item] += 1;
			else:
				freq[item] = 1;


for key, value in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
	#fw.write(key+"	"+str(value)+"\n")
	fw.write(key+"\n")