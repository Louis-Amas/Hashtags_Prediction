import numpy as np
from sys import stdout

def to_one_hot(car, vocab):
	one_hot_vec = np.zeros(len(vocab), dtype=np.bool)
	one_hot_vec[vocab[car]] = 1
	return one_hot_vec

def one_hot_to_char(one_hot_vec, rev_vocab):
	return rev_vocab[np.argmax(one_hot_vec)]

def process_text_to_char(texts, vocab_dic, seq_len):
	'''
		Transform text to one hot vector
	'''
	processed_text = np.zeros((len(texts), seq_len, len(vocab_dic)), dtype=np.bool)
	for i, text in enumerate(texts):
		max_iter = min(seq_len, len(text))

		for j in range(max_iter):
			processed_text[i][j] = to_one_hot(text[j], vocab_dic)


	return processed_text

def vec_to_one_hot_matrix(vec, vocab):
	one_hot_mat = np.zeros((len(vec), len(vocab)))
	for i, word in enumerate(vec):
		one_hot_mat[i] = to_one_hot(word, vocab)
	return one_hot_mat


def process_label(texts, vocab_label):
	labels = np.zeros(len(texts))
	for i, label in enumerate(texts):
	    labels[i] = vocab_label[label[0]]
	return labels


def process_text_to_normal(processed_text, rev_vocab):
	tab = list()
	for text in processed_text:
		seq = ''
		for car in text:
			seq += one_hot_to_char(car, rev_vocab)
		tab.append(seq)
	return tab

def create_char_vocab(texts):
	vocab_set = set()
	for text in texts:
		for car in text:
			vocab_set.add(car)

	vocab_dic = dict()
	vocab_dic['<eos>'] = 0
	for i, car in enumerate(vocab_set):
		vocab_dic[car] = i+1
	
	rev_vocab = {vocab_dic[key]: key for key in vocab_dic}

	return vocab_dic, rev_vocab


def create_word_vocab(texts):
	words_set = set()
	for word in texts:
		words_set.add(word[0])
	words_dic = { word: i for i, word in enumerate(words_set)}
	rev_words = { words_dic[word]: word for word in words_dic}
	return words_dic, rev_words


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
	'''
		Examples
	'''
	texts = ['hi how aaaae u',
		 'hi how r u'
		]

	vocab_dic, rev_vocab = create_char_vocab(texts)
	p_texts = process_text_to_char(texts, vocab_dic, 15)
	print(process_text_to_normal(p_texts, rev_vocab))