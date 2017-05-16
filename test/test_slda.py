import numpy as np

from LDA import slda
from LDA.utils import make_docs

np.random.seed(0)
K = 3
V = 5
alpha = 1 / K
train_corpus = [np.array([1, 2, 0, 4, 1, 1]), np.array([3, 0, 3, 1]), np.array([2, 4, 2, 4]), np.array([2, 3, 1])]
train_resps = np.log(np.array([1.2, 3, 2, 2.5]))    # assume log normality; see bottom of p.7 of sLDA paper
train_docs = make_docs(train_corpus)

# learn the parameters by variational EM
Beta, mu, var = slda.vem(docs=train_docs, resps=train_resps, alpha=alpha, K=K, V=V, niter=10)

print("test new documents")
test_corpus = [np.array([2, 1, 1, 0, 4]), np.array([2, 3, 3])]
test_docs = make_docs(test_corpus)
print(np.exp(slda.predict(test_docs, alpha, K, Beta, mu)))
