import numpy as np
from nltk.corpus import reuters

import LDA.utils as utils
from LDA import lda

# add all training documents' names to a list
num_docs = 100  # use the first `num_docs` for training
train_list = reuters.fileids()[:num_docs]

# pre-processing; implicitly determine the vocabulary
words_list = [reuters.words(fid) for fid in train_list]  # can skip tokenization as nltk did it already
# word level preprocessing
words_list = [utils.pre_process_text(words, remove_stop=True, remove_punc=True, remove_num=False, min_len=2,
                                     lower_case=True, stem=False) for words in words_list]
# document/corpus level preprocessing
words_list = utils.remove_outlier_terms(words_list, min_idc=5, max_idc=None, min_idf=None, max_idf=0.2)

# make sure to get rid of empty documents as a result of possibly aggressive pre-processing
min_doc_length = 5
words_list = [words for words in words_list if len(words) > min_doc_length]
num_docs = len(words_list)

# done pre-processing; convert to integers
word2int, int2word = utils.map_word_int(words_list)

ints_list = utils.words_list_to_ints_list(words_list, word2int)  # list of integer arrays

docs = utils.make_docs(ints_list)

print('V =', len(word2int))

# run the learning procedure (Gibbs sampling and Variational)
np.random.seed(0)
K = 10
V = len(word2int)
alpha = 0.1
eta = 0.1

# Gamma, Beta, Phi = LDA.vem(docs=docs, K=K, V=V, alpha=alpha, niter=5)
# # get an idea of the most common words under each topic
# n = 10  # top n
# for lamb in Beta:
#     idx = np.argpartition(lamb, -n)[-n:]
#     idx = reversed(idx[np.argsort(lamb[idx])])
#     print([(int2word[i], lamb[i]) for i in idx])

Gamma, Lambda, Phi = lda.vi(docs=docs, K=K, V=V, alpha=alpha, eta=eta, tol=0.001)
# get an idea of the most common words under each topic
n = 10  # top n
for lamb in Lambda:
    idx = np.argpartition(lamb, -n)[-n:]
    idx = reversed(idx[np.argsort(lamb[idx])])
    print([(int2word[i], lamb[i]) for i in idx])
    print()
