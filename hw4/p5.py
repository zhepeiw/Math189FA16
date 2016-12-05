import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def NMF(V, r, max_iter=200, tolerance=1e-3, print_frq = 20):
	n = V.shape[0]
	m = V.shape[1]
	obj = []

	def dist(V, U):
		assert(V.shape == U.shape)	
		return np.linalg.norm(V - U)
		
	def genW(row, col, W, VHt, WHHt):
		return W.item(row, col) * (VHt.item(row, col)) / (WHHt.item(row, col))

	def genH(row, col, H, WtV, WtWH):
		return H.item(row, col) * (WtV.item(row, col)) / (WtWH.item(row, col))
	# W: n x r, H: r x m
	W, H = np.mean(V) * np.matrix(np.random.rand(n, r)), np.mean(V) * np.matrix(np.random.rand(r, m))
	for iteration in range(max_iter):
		if iteration % print_frq == 0 and iteration != 0:
			print("[iter={}] objective={}".format(iteration, obj[-1]))
		VHt = V * H.transpose()
		WHHt = W * H * H.transpose()
		W = np.matrix([[genW(row, col, W, VHt, WHHt) for col in range(r)] for row in range(n)])

		WtV = W.transpose() * V
		WtWH = W.transpose() * W * H
		H = np.matrix([[genH(row, col, H, WtV, WtWH) for col in range(m)] for row in range(r)])

		d = dist(V, W * H)
		obj.append(d)
		if d <= tolerance:
			break
	
	return W, H, obj



def genCvgPlot(obj):
	plt.style.use('bmh')
	plt.figure(1)
	plt.title('NMF Convergence Plot')
	plt.xlabel('# Iteration')
	plt.ylabel('RMSE')
	plt.plot(obj, 'r')
	plt.savefig('cvg_5.pdf', format='pdf')
	plt.show()

def genFile():
	dirname = "reuters21578"
	filenames = sorted([os.path.join(dirname, fn) for fn in os.listdir(dirname) if fn.endswith(".sgm")])
	output_dir = "samples"

	count = 0

	def save_content(content):
		global count
		name = os.path.join(output_dir, "samples-" + str(count) + ".txt")
		count += 1
		with open(name, 'w') as fn:
			fn.write(content)

	def parse_file(content):
		i = 0
		ans = []
		while i < len(content):
			j = i
			while j < len(content) and content[j] != '<':
				j += 1
			if j + 6 <= len(content) and content[j:j + 6] == "<BODY>":
				k = j + 6
				while k < len(content) and content[k] != '<':
					k += 1
				if k + 7 <= len(content) and content[k:k + 7] == "</BODY>":
					# Get rid of "\n&#3;"
					e = k - 5
					# Get rid of "\n Reuter"				
					if content[e - 6:e].lower() == "reuter":
						e -= 8
					save_content(content[j + 6:e])
				j = k + 7
			i = j + 1
	# Create output directory
	if not os.path.exists(output_dir):
	    os.makedirs(output_dir)
	else:
		shutil.rmtree(output_dir)

	for fn in filenames:
		with open(fn) as fd:
			parse_file(fd.read())

# process txt files
CORPUS_PATH = os.path.join('samples')
num_topics = 15
num_top_words = 20
filenames = sorted([os.path.join(CORPUS_PATH, fn) for fn in os.listdir(CORPUS_PATH)][:1000])
import sklearn.feature_extraction.text as text
vectorizer = text.CountVectorizer(input='filename', stop_words='english', decode_error='ignore', min_df=5)
dtm = vectorizer.fit_transform(filenames).toarray()
# print dtm.shape
vocab = np.array(vectorizer.get_feature_names())

def generateReport():
	W, H, obj = NMF(dtm, num_topics)
	H = np.array(H)
	# print dtm.shape, H.shape
	topic_words = []
	for topic in H:
		word_idx = np.argsort(topic)[::-1][0:num_top_words]
		topic_words.append([vocab[i] for i in word_idx])
	for t in range(len(topic_words)):
		print("Topic {}: {}".format(t + 1, ' '.join(topic_words[t][:num_top_words])))
	genCvgPlot(obj)

generateReport()

