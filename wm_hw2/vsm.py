import sys
import numpy as np
import pandas as pd
import json
import math

### parameters
ka = 1000
k1 = 2
b = 0.75

terms = {}
data = []

### load inverted file
with open(sys.argv[1]) as f:
	dict0 = json.load(f)
	i = 0
	for (word, dict1) in dict0.items():
		terms[word] = i
		data.append(dict1['docs'])
		i += 1

news_num = 0
news_length = []
avg_length = 0

### load news file
with open(sys.argv[2]) as f:
	news = json.load(f)
	news_num = len(news)
	for (url, content) in news.items():
		news_length.append(len(content))
		avg_length += len(content)
	avg_length /= news_num

### load query terms
query_num = 20
query_term = [[] for _ in range(query_num)]
query_vec = [[] for _ in range(query_num)]
def get_query():
	df_query = pd.read_csv(sys.argv[3])
	query = [row for row in df_query['Query'].tolist()]
	for (n, q) in enumerate(query):
		for i in range(len(q)-1):
			if i+3 < len(q):
				if q[i:i+4] in terms:
					query_term[n].append(terms[q[i:i+4]])
					query_vec[n].append(1)
			if i+2 < len(q):
				if q[i:i+3] in terms:
					query_term[n].append(terms[q[i:i+3]])
					query_vec[n].append(1)
			if q[i:i+2] in terms:
				query_term[n].append(terms[q[i:i+2]])
				query_vec[n].append(1)


def Okapi(c, length, k1, b):
	return ((k1 + 1) * count) / (k1 * (1 - b + b*(length/avg_length)) + count)

def get_answer(query, weight):
	tmp_ans = []
	for n in range(news_num):
		if np.linalg.norm(weight[n]) == 0:
			continue
		score = np.dot(query, weight[n])
		if score > 0:
			tmp_ans.append([n, score])
	tmp_ans = sorted(tmp_ans, key=lambda x:x[1], reverse=True)
	if len(tmp_ans) >= 300:
		tmp_ans = tmp_ans[:300]
	for i in range(len(tmp_ans)):
		tmp_ans[i][0] = 'news_' + '{0:06}'.format(tmp_ans[i][0]+1)
	return tmp_ans

alpha, belta = 1, 0.2
def relevance_feedback(query, weight, tmp_ans):
	rel_ans = tmp_ans[:10]
	original = alpha * query
	### rel
	rel_vec = np.zeros(shape=(query.shape))
	for i in range(10):
		news_id = int(rel_ans[i][0][5:])-1
		rel_vec += weight[news_id]
	rel_vec = (belta / 10) * rel_vec
	return original + rel_vec

ans = []
get_query()
for i in range(query_num):
	weight = []
	for t in query_term[i]:
		tmp = [0 for _ in range(news_num)]
		idf = math.log((news_num - len(data[t]) + 0.5) / (len(data[t]) + 0.5))
		for dic in data[t]:
			tmp_list = tuple(dic.items())
			news_id, count = tmp_list[0][0], tmp_list[0][1]
			news_id = int(news_id[5:]) - 1
			#print(news_id)
			tmp[news_id] = Okapi(count, news_length[news_id], k1, b) * idf
		weight.append(tmp)
	weight = np.array(weight).transpose()
	query = np.array(query_vec[i])
	for j in range(len(query)):
		query[j] = (ka+1)*query[j] / (ka + query[j])

	tmp_ans = get_answer(query, weight)

	### relevance feedback
	query = relevance_feedback(query, weight, tmp_ans)
	tmp_ans = get_answer(query, weight)

	ans.append([a[0] for a in tmp_ans])

### write answer
template = pd.read_csv(sys.argv[4])
for i in range(300):
	rank = "Rank_" + '{0:03}'.format(i+1)
	template[rank] = [ans[n][i] for n in range(query_num)]
template.to_csv(sys.argv[5], index=None)
	