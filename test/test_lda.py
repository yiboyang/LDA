import numpy as np

from LDA import lda
from LDA.utils import make_docs

np.random.seed(0)
K = 3
V = 5
alpha = 1 / K
train_corpus = [np.array([1, 2, 0, 4, 1, 1]), np.array([3, 0, 3, 1]), np.array([2, 4, 2, 4]), np.array([2, 3, 1])]
train_docs = make_docs(train_corpus)


#lda.cgs(docs=train_docs, K=K, V=V, alpha=0.1, eta=1, niter=20, debug=True)
#lda.vem(docs=train_docs, alpha=1/K, K=K, V=V, niter=10)
lda.vi(docs=train_docs, K=K, V=V, alpha=alpha, eta=0.01)
