import io

#fr = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train.txt',  'r' , encoding="utf8")
#fw = open('D:\Dropbox\ML\DSL-Task-master\data\DSLCC-v2.0\\train-dev\\train200.txt',  'w' , encoding="utf8")

fr = io.open('/home/nirob/Dropbox/ML/DSL-Task-master/data/DSLCC-v2.0/train-dev/train.txt', 'r', encoding="utf8")
fw = io.open('/home/nirob/Dropbox/ML/DSL-Task-master/data/DSLCC-v2.0/train-dev/train1000.txt', 'w', encoding="utf8")

count = 0
lang = "bg"

for line in fr:
	word_list = line.split()
	if word_list[-1] != lang:
		lang = word_list[-1]
		count = 0
	count += 1
	if count <= 1000:
		fw.write(line)


fr.close()
fw.close()