fr = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train.txt',  'r' , encoding="utf8")
fw = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train2.txt',  'w' , encoding="utf8")


for line in fr:
	word_list = line.split()
	fw.write(word_list[-1]+'\n')


fr.close()
fw.close()