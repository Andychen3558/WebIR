import xml.etree.cElementTree as ET
import math
import numpy as np
import sys

start = 2
has_feedback = 0
if sys.argv[1] == '-r':
	start += 1
	has_feedback = 1
elif sys.argv[1] == '-b':
	start += 1

vocab = {}
file = []
doc_length = []
avg_length = 0
bigram = {}
data = []

### parameters
ka = 1000
k1 = 2
b = 0.75
alpha, belta, gamma = 0.95, 0.05, 0.01

with open(sys.argv[start+4] + "vocab.all", 'r') as f:
	i = 0
	for voc in f:
		voc = voc.strip()
		vocab[voc] = i
		i += 1

with open(sys.argv[start+4] + "file-list", 'r') as f:
	i = 0
	for news in f:
		news = news.strip().split('/', 1)[1]
		file.append(news.split('/')[2].lower())
		### calculate document length
		doc = ET.ElementTree(file=sys.argv[start+6] + '/' + news)
		root_doc = doc.getroot()
		length = 0
		for text in root_doc.iter('doc'):
			for p in text.iter('p'):
				length += len(p.text)
		doc_length.append(length)
		avg_length += length

		i += 1
	total_file = i
	avg_length /= total_file

with open(sys.argv[start+4] + "inverted-file", 'r') as f:
	term_count = 0
	while True:
		line = f.readline()
		if not line:
			break
		line = line.strip().split(' ')
		v1, v2, N = int(line[0]), int(line[1]), int(line[2])
		if v2 == -1:
			for i in range(N):
				f.readline()
			continue
		line[0] += (','+line[1])
		bigram[line[0]] = term_count
		term_count += 1

		tmp = []
		for i in range(N):
			line1 = f.readline().strip().split(' ')
			file_id, count = int(line1[0]), int(line1[1])
			tmp.append([file_id, count])
		data.append(tmp)

tree = ET.ElementTree(file=sys.argv[start])
root = tree.getroot()
length = len(root.getchildren())
query_term = [[] for _ in range(length)]
query_vec = [[] for _ in range(length)]
def get_term(tag):
	now = 0
	for child in root.iter(tag):
		cur = child.text.strip()
		cur = cur.split('。')[0]
		for i in range(len(cur)-1):
			if cur[i] in vocab and cur[i+1] in vocab:
				v1, v2 = vocab[cur[i]], vocab[cur[i+1]]
				gram = str(v1) + ',' + str(v2)
				if gram not in bigram:
					continue
				term = bigram[gram]
				if term in query_term[now]:
					index = query_term[now].index(term)
					query_vec[now][index] += 1
					continue
				query_term[now].append(term)
				query_vec[now].append(1)
		now += 1

def Okapi(c, length, k1, b):
	return ((k1 + 1) * count) / (k1 * (1 - b + b*(length/avg_length)) + count)

def get_answer(query, weight):
	tmp_ans = []
	for n in range(total_file):
		if np.linalg.norm(weight[n]) == 0:
			continue
		score = np.dot(query, weight[n])
		if score > 0:
			tmp_ans.append([n, score])
	tmp_ans = sorted(tmp_ans, key=lambda x:x[1], reverse=True)
	if len(tmp_ans) >= 100:
		tmp_ans = tmp_ans[:100]
	return tmp_ans

def relevance_feedback(query, weight, tmp_ans):
	rel_ans = tmp_ans[:40]
	irrel_ans = tmp_ans[40:]
	original = alpha * query
	### rel
	rel_vec = np.zeros(shape=(query.shape))
	for i in range(40):
		rel_vec += weight[rel_ans[i][0]]
	rel_vec = (belta / 40) * rel_vec
	### non-rel
	irrel_vec = np.zeros(shape=(query.shape))
	for i in range(60):
		irrel_vec += weight[irrel_ans[i][0]]
	irrel_vec = (gamma / 60) * irrel_vec
	return original + rel_vec + irrel_vec


if __name__ == '__main__':
	get_term("concepts")

	f = open(sys.argv[start+2], 'w')
	f.write("query_id,retrieved_docs\n")

	ans = []
	Map = 0.0
	for i in range(length):
		weight = []
		for term in query_term[i]:
			tmp = [0 for _ in range(total_file)]
			#idf = math.log(total_file / len(data[term]))
			idf = math.log((total_file - len(data[term]) + 0.5) / (len(data[term]) + 0.5))
			for (file_id, count) in data[term]:
				tmp[file_id] = Okapi(count, doc_length[file_id], k1, b) * idf
			weight.append(tmp)
		weight = np.array(weight).transpose()
		query = np.array(query_vec[i])
		for j in range(len(query)):
			query[j] = (ka+1)*query[j] / (ka + query[j])

		tmp_ans = get_answer(query, weight)

		### relevance feedback
		if has_feedback == 1:
			query = relevance_feedback(query, weight, tmp_ans)
			tmp_ans = get_answer(query, weight)

		ans.append(tmp_ans)

	### write answer
	i = 0
	for child in root.iter("number"):
		index = child.text.strip()[-3:] + ','
		f.write(index)
		line = ""
		for (n, score) in ans[i]:
			line += (file[n] + ' ')
		line = line[:-1] + '\n'
		f.write(line)
		i += 1
		
		