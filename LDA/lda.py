"""
Latent Dirichlet Allocation main model
Currently we use variational EM as in the original paper http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf,
but for simplicity treat alpha as a known scalar (hyperparameter of a symmetric Dirichlet prior) that is not optimized
in the variational M-step.

The variational E-step for a full Bayesian treatment (see method `vi`) has been implemented following
http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixed-membership.pdf, but an M-step has yet to be written to optimize
over alpha and eta (alternatively they can be taken to be known constants, i.e. the user needs to specify these
hyperparameters, then there's no need for an M step)
"""

import numpy as np
import scipy.optimize
import scipy.stats
from scipy.special import gammaln, digamma


class Document:
    """
    An object keeping track of words in a document; modeled by a categorical distribution over topics in LDA
    """

    def __init__(self, words, beg_w_pos=None):
        """

        :param words: a numpy array of integers in [0,...,V-1], where V is the vocab size
        :param beg_w_pos: the position/index within the corpus of the first word in document
        """
        self.words = words
        self.len = len(words)
        self.beg_w_pos = beg_w_pos
        # for convenience; the last word index is actually end_w_pos-1, so that all_words[beg_w_pos:end_w_pos] correctly
        # selects words in the document
        self.end_w_pos = beg_w_pos + self.len
        self.assignment = self.count_topics = None

    def update_assignment(self, assignment, K):
        """
        Update current topic assignment of words to a new one
        :param assignment: an assignment of words to topics, a list of ints in [0,...,K-1]
        :param K: total number of topics
        :return: None
        """
        self.assignment = assignment
        self.count_topics = np.bincount(self.assignment, minlength=K)  # count of number of words under each topic

    def sample_assignment(self, theta, update=False):
        """
        Sample a topic assignment to all the words in this document, given its categorical distribution over topics
        :param theta: length-K vector of categorical distribution parameters over topics
        :param update: whether to update current assignment with the new sample
        :return: sample topic assignment, a vector of length len(words) in range(K)
        """
        sample = np.random.choice(len(theta), size=self.len, p=theta)
        if update:
            self.update_assignment(sample, len(theta))
        return sample


class Topic:
    """
    An object keeping track of words belonging to this topic; modeled by a categorical distribution over the vocabulary
    in LDA.
    """

    def __init__(self, idx=None):
        """
        An object keeping track of words belonging to this topic
        :param idx: int, a numeric index/name for this topic to help debug
        """
        self.idx = idx
        self.counter = self.total_count = None

    def update_count(self, counter):
        """
        Update the Counter object and total_count
        :param counter: a Counter for the counts of words in the corpus belonging to this topic
        :return:
        """
        self.counter = counter
        self.total_count = sum(self.counter.values())  # total number of words under this topic

    def __repr__(self):
        return "%s idx=%d" % (self.__class__, self.idx)


# methods for LDA

# sampling stuff
def cgs(docs, K, V, alpha, eta, niter, debug=True):
    """
    Collapsed Gibbs sampling; for efficiency we don't explicitly maintain the parameters except in debug mode
    :param docs: document objects whose distributions are to be approximated
    :param K: # topics
    :param V: vocab size
    :param alpha: float, symmetric Dirichlet prior hyperparam for the topic mixing proportion for each document
    :param eta: float, symmetric Dirichlet prior hyperparam for the word distribution for each topic
    :param niter: # of iterations
    :param debug: if enabled, will sample new param for each iteration and calculate/print log joint
    :return:
    """
    # for counting
    from collections import Counter
    from functools import reduce
    from operator import add
    if debug:
        from . import utils

    D = len(docs)
    topics = [Topic(idx=k) for k in range(K)]

    if debug:
        # Dirichlet hyperparameter matrices; not explicitly used in cgs
        Gamma = np.empty((D, K))
        Lambda = np.empty((K, V))

        # parameter matrices; we draw them from Dirichlet priors
        Theta = np.empty_like(Gamma)
        Beta = np.empty_like(Lambda)

    # sample an initial assignment of all words to topics; draw assignments using theta samples from priors
    for j, d in enumerate(docs):
        d.sample_assignment(theta=scipy.stats.dirichlet.rvs(alpha=np.ones(K) * alpha)[0], update=True)

    # count the number of words by topics across corpus
    for k in range(K):  # TODO: parallelize if needed
        count_terms_by_topic = reduce(add, (Counter(d.words[d.assignment == k]) for d in docs))
        topics[k].update_count(count_terms_by_topic)

    for it in range(niter):
        # go through the assignment of every word in corpus
        for i, d in enumerate(docs):
            for j, w, a in zip(range(d.len), d.words, d.assignment):
                # remove current word assignment
                d.count_topics[a] -= 1
                topics[a].counter[w] -= 1
                topics[a].total_count -= 1

                # calculate the probability vector from which to sample; it's a product of Dirichlet expectations
                expected_theta_d = (alpha + d.count_topics) / (alpha * K + d.len)
                expected_beta_w = [(eta + topics[k].counter[w]) /
                                   (eta * V + topics[k].total_count) for k in range(K)]

                prob = expected_theta_d * expected_beta_w
                prob /= sum(prob)
                k = np.random.choice(K, p=prob)  # sample a new topic assignment

                # put new assignment back in corpus
                d.assignment[j] = k
                d.count_topics[k] += 1
                topics[k].counter[w] += 1
                topics[k].total_count += 1

        if debug:
            # get parameter estimates using samples to get a rough idea of log likelihood
            for i, d in enumerate(docs):
                Gamma[i] = alpha + d.count_topics
                Theta[i] = np.random.dirichlet(Gamma[i])  # sample a new theta
            for k, t in enumerate(topics):
                Lambda[k] = eta + utils.counter_to_vec(t.counter, length=V)
                Beta[k] = np.random.dirichlet(Lambda[k])  # sample a new beta

            # print([str(d.assignment) for d in docs])
            print(log_joint(docs, topics, K, Gamma, Lambda, Theta, Beta))  # should roughly be increasing


def log_data_likelihood(docs, topics, Gamma, Theta, Beta):
    """
    Compute log probability of complete data (document mixing proportions, words, and topic assignments) conditioned
    on mixture components parameters (categorical dists over topics)
    :param Gamma: DxK Dirichlet hyperparam over documents (would be np.ones((D,K))*alpha for the prior)
    :param Theta: DxK categorical parameters of documents
    :param Beta: KxV categorical parameters of topics
    :return: float
    """
    result = 0

    for i, d in enumerate(docs):
        result += scipy.stats.dirichlet.logpdf(Theta[i], Gamma[i])  # log p(theta_d)
        result += np.dot(np.log(Theta[i]), d.count_topics)  # sum_{n=1}^N_d log p(z_{dn}|theta_d)

    for k, t in enumerate(topics):
        c = t.counter
        if len(c) == 0:
            continue
        terms, counts = zip(*c.items())
        result += np.dot(np.log(Beta[k][list(terms)]),
                         list(counts))  # sum_{d=1}^D sum_{n=1}^N_d log p(w_{dn}|z_{dn}, betas, theta_d)

    return result


def log_comp_params(K, Lambda, Beta):
    """
    Compute log probability of mixture components (topics) parameters under current model
    :param Lambda: KxV Dirichlet hyperparam over topics (would be np.ones((K,V))*eta for the prior)
    :param Beta: KxV categorical parameters of topics
    :return:  float
    """
    return sum(scipy.stats.dirichlet.logpdf(Beta[k], Lambda[k]) for k in
               range(K))  # sum_{k=1}^K log p(beta_k)


def log_joint(docs, topics, K, Gamma, Lambda, Theta, Beta):
    """
    Compute the log probability of the LDA joint distribution
    :return: float
    """
    return log_data_likelihood(docs, topics, Gamma, Theta, Beta) + log_comp_params(K, Lambda, Beta)


# variational stuff
def vem_estep(docs, K, alpha, Beta, prev_params=None, tol=0.001, max_try=20):
    """
    Variational Bayesian inference for LDA. We adopt a semi-Bayesian approach and maintain independent variational
    distributions over the latent variables of the model, thetas and zs.
        Gamma: D x K matrix of parameters for the variational Dirichlet distribution for all the documents, where
            the dth row parametrizes document d, a Dirichlet distribution over topics
        Phi: W x K matrix parameters for the variational categorical distribution for all the topic assignments,
            where W is the total number of words in corpus, and the wth row encodes a probability vector over the
            topic assignment for word w.
    All the expectations in the code are taken wrt to the variational distribution q. Equations referenced are from
    the LDA paper http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf
    :param docs: document objects whose distributions are to be approximated
    :param K: # topics
    :param alpha: float, symmetric Dirichlet prior hyperparam for the topic mixing proportion for each document
    :param Beta: 2d array, KxV categorical parameters for topics
    :param prev_params: variational parameters from previous iteration, (Gamma, Phi); if provided will generally
        make vem more efficient
    :param tol: float, error tolerance, the minimum percentage increase of elbo, below which the algorithm is
        considered to have converged
    :param max_try: int; if elbo has not increased by more than tol for max_try iterations, return
    :return:
    """

    def elbo(E_log_Theta, E_log_p_zw, Gamma, Phi):
        """
        Calculate the variational lower bound on log evidence. Uses intermediate calculations in vi for efficiency.
        We use the equation L = E_q[log p(X,Z)] + H[log q(Z)], where H is the entropy; for LDA it can be decomposed
        L = E_q[log p(theta|alpha)] + E_q[log p(z,w|theta,beta)] + H[q(theta|gamma)] + H[q(z|phi)]
        The unspecified parameters are constant global variables.
        :return: variational lower bound for naive mean field
        """
        E_log_p_theta = (alpha - 1) * E_log_Theta.sum() + D * (gammaln(alpha * K) -
                                                               K * gammaln(alpha))  # line 1 of eq (15)

        H_q_theta = sum(scipy.stats.dirichlet.entropy(g) for g in Gamma)

        H_q_z = -(Phi * np.log(Phi)).sum()

        lb = E_log_p_theta + E_log_p_zw + H_q_theta + H_q_z
        return lb

    D = len(docs)
    W = sum(d.len for d in docs)  # total # of words in corpus

    if prev_params is None:
        # random initialization based on Blei's paper figure 6
        Gamma = np.random.rand(D, K) + alpha + np.array([d.len / K for d in docs])[:, np.newaxis]
        Phi = np.random.dirichlet(alpha=np.ones(K) * alpha, size=W)
    else:
        Gamma, Phi = prev_params

    lb_prev = float("inf")  # elbo from previous iteration

    while True:
        # do some preliminary calculations for lower bound computation as well as parameter updates;
        # try to operate on large matrices for efficiency on PC; may take forever :)
        E_log_Theta = digamma(Gamma) - digamma(Gamma.sum(axis=1))[:, np.newaxis]  # eq (8), for all thetas
        log_Beta = np.log(Beta + 1e-32)  # for numeric stability

        E_log_p_zw = 0  # one of the terms in elbo that can be nicely computed here
        log_Phi_new = np.zeros_like(Phi)  # the RHS of (6), equal to log of the new Phi plus a constant
        for i, d in enumerate(docs):
            log_Phi_new[d.beg_w_pos: d.end_w_pos] += E_log_Theta[i] + log_Beta[:, d.words].T  # eq (6)
            E_log_p_zw += np.sum(log_Phi_new[d.beg_w_pos: d.end_w_pos] *
                                 Phi[d.beg_w_pos: d.end_w_pos])  # line 2,3 of eq (15) combined

        # at this point no parameters have actually been updated yet; we invoke elbo() to calculate the lower bound
        # of variational distribution from the previous iteration
        lb = elbo(E_log_Theta, E_log_p_zw, Gamma, Phi)
        print(lb)
        if lb_prev != float("inf") and abs((lb - lb_prev) / lb_prev) < tol:  # if no improvement in elbo
            if num_try > 0:
                num_try -= 1
            else:  # num_try == 0
                break
        else:  # if there was improvement in elbo
            num_try = max_try
            lb_prev = lb

        # update distributions phi_dn over topic assignments for all words
        # take exp and normalize
        Phi = np.exp(log_Phi_new)
        Phi /= Phi.sum(axis=1)[:, np.newaxis]  # (6) complete

        # update distributions gamma_d over documents
        for i, d in enumerate(docs):
            Gamma[i] = alpha + np.sum(Phi[d.beg_w_pos: d.end_w_pos], axis=0)  # eq (7)

    return Gamma, Phi


def vem_mstep(docs, K, V, Phi):
    """
    M-step of variational EM to estimate hyperparameters using maximum likelihood based on expected sufficient stats
    under approximate posterior; same as maximizing ELBO wrt to Beta
    :return:
    """
    # D = len(docs)
    # # MLE for alpha; since we use a symmetric Dirichlet distribution, alpha is just a scalar
    # suff_stats = E_log_Theta.sum()
    #
    # def dLda(alpha):    # first derivative of lower bound wrt to alpha
    #     return D * K *(digamma(K * alpha) - digamma(alpha)) + suff_stats
    #
    # def dd(alpha):  # second derivative of lower bound wrt to alpha
    #     return D * K * K * polygamma(1, K * alpha) - D * K * polygamma(1, alpha)
    #
    # alpha = scipy.optimize.newton(func=dLda, x0=1, fprime=dd)  # scalar version of appendix A.4.2
    # assert alpha > 0
    # # no guarantee the above is correct; alpha < 0 indicates trouble (possible severe overfitting); try different
    # # settings of x0 until you get something reasonable (the optimization problem doesn't seem to be convex at all)


    Beta = np.zeros((K, V))  # categorical param for topics
    for j, w in enumerate(w for d in docs for w in d.words):  # loop through all words
        Beta[:, w] += Phi[j]
    Beta /= np.sum(Beta, axis=1)[:, np.newaxis]  # eq (9), MLE for Beta

    return Beta


def vem(docs, alpha, K, V, niter):
    """
    Convenience method for learning an LDA model parameters with variational EM. Currently not learning alpha
    because it is slightly complicated; it needs to be provided as a fixed hyperparameter.
    :param docs:
    :param alpha:
    :param K:
    :param V:
    :param niter:
    :return:
    """
    Beta = np.random.dirichlet(np.ones(V), K)  # model params

    # variational params
    Gamma = np.random.rand(len(docs), K) + alpha + np.array([d.len / K for d in docs])[:, np.newaxis]
    Phi = np.random.dirichlet(alpha=np.ones(K) * alpha, size=sum(d.len for d in docs))
    for it in range(niter):
        Gamma, Phi = vem_estep(docs, K, alpha, Beta, prev_params=(Gamma, Phi), tol=0.1, max_try=5)
        Beta = vem_mstep(docs, K, V, Phi)  # optimized model parameters

    return Gamma, Beta, Phi


def vi(docs, K, V, alpha, eta, tol=0.001, max_try=20):
    """
    Variational Bayesian inference for LDA. We adopt a full Bayesian approach and maintain independent variational
    distributions over all the unknown variables of the model:
        Lambda: K x V matrix of parameters for the variational Dirichlet distribution for all the topics, where the
            kth row parametrizes topic k, a Dirichlet distribution over terms in vocab
        Gamma: D x K matrix of parameters for the variational Dirichlet distribution for all the documents, where
            the dth row parametrizes document d, a Dirichlet distribution over topics
        Phi: W x K matrix parameters for the variational categorical distribution for all the topic assignments,
            where W is the total number of words in corpus, and the wth row encodes a probability vector over the
            topic assignment for word w.
    All the expectations in the code are taken wrt to the variational distribution q. Equations referenced are from
    Blei's tutorial http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixed-membership.pdf
    :param docs: document objects whose distributions are to be approximated
    :param K: # topics
    :param V: vocab size
    :param alpha: float, symmetric Dirichlet prior hyperparam for the topic mixing proportion for each document
    :param eta: float, symmetric Dirichlet prior hyperparam for the word distribution for each topic
    :param tol: float, error tolerance, the minimum percentage increase of elbo, below which the algorithm is
        considered to have converged
    :param max_try: int; if elbo has not increased by more than tol for max_try iterations, return
    :return:
    """
    D = len(docs)
    W = sum(d.len for d in docs)  # total # of words in corpus

    def elbo(E_log_Theta, E_log_Beta, E_log_p_zw, Gamma, Lambda, Phi):
        """
        Calculate the variational lower bound on log evidence. Uses intermediate calculations in vi for efficiency.
        We use the equation L = E_q[log p(X,Z)] + H[log q(Z)], where H is the entropy; for LDA it can be decomposed
        L = E_q[log p(theta|alpha)] + E_q[log p(beta|eta)] + E_q[log p(z,w|theta,beta)]
        H[q(theta|gamma)] + H[beta|lambda)] + H[q(z|phi)]
        :return: variational lower bound for naive mean field
        """
        E_log_p_Theta = (alpha - 1) * E_log_Theta.sum() + D * (gammaln(alpha * K) -
                                                               K * gammaln(alpha))
        E_log_p_Beta = (eta - 1) * E_log_Beta.sum() + K * (gammaln(eta * V) -
                                                           V * gammaln(eta))

        H_q_theta = sum(scipy.stats.dirichlet.entropy(g) for g in Gamma)
        H_q_beta = sum(scipy.stats.dirichlet.entropy(l) for l in Lambda)
        H_q_z = -(Phi * np.log(Phi)).sum()

        lb = E_log_p_Theta + E_log_p_zw + E_log_p_Beta + H_q_theta + H_q_z + H_q_beta
        return lb

    lb_prev = float("inf")  # elbo from previous iteration
    num_try = max_try  # number of tries left before returning

    # random initialization based on Blei's paper figure 6
    Gamma = np.random.rand(D, K) + alpha + np.array([d.len / K for d in docs])[:, np.newaxis]
    Lambda = np.random.rand(K, V) + eta + W / V
    Phi = np.random.dirichlet(alpha=np.ones(K) * alpha, size=W)

    while True:
        # do some preliminary calculations for lower bound computation as well as parameter updates;
        # try to operate on large matrices for efficiency on PC; may take forever :)
        E_log_Theta = digamma(Gamma) - digamma(Gamma.sum(axis=1))[:, np.newaxis]  # eq (47), for all thetas
        E_log_Beta = digamma(Lambda) - digamma(Lambda.sum(axis=1))[:, np.newaxis]  # eq (48), for all betas

        # calculate expectations in eq (46)
        E_log_p_zw = 0  # one of the terms in elbo that can be nicely computed here
        log_Phi_new = np.zeros_like(Phi)  # the RHS of (46), equal to log of the new Phi plus a constant
        for i, d in enumerate(docs):
            log_Phi_new[d.beg_w_pos: d.end_w_pos] += E_log_Theta[i]  # eq (47) for all words
            log_Phi_new[d.beg_w_pos: d.end_w_pos] += E_log_Beta[:, d.words].T  # eq (48) for all words
            E_log_p_zw += np.sum(log_Phi_new[d.beg_w_pos: d.end_w_pos] *
                                 Phi[d.beg_w_pos: d.end_w_pos])  # eq (46) weighted by Phi

        # at this point no parameters have actually been updated yet; we invoke elbo() to calculate the lower bound
        # of variational distribution from the previous iteration
        lb = elbo(E_log_Theta, E_log_Beta, E_log_p_zw, Gamma, Lambda, Phi)
        print(lb)
        if lb_prev != float("inf") and abs((lb - lb_prev) / lb_prev) < tol:  # if no improvement in elbo
            if num_try > 0:
                num_try -= 1
            else:  # num_try == 0
                break
        else:  # if there was improvement in elbo
            num_try = max_try
            lb_prev = lb

        # update distributions phi_dn over topic assignments for all words
        # take exp and normalize
        Phi = np.exp(log_Phi_new)
        Phi /= Phi.sum(axis=1)[:, np.newaxis]  # (46) complete

        # update distributions gamma_d over documents
        for i, d in enumerate(docs):
            Gamma[i] = alpha + np.sum(Phi[d.beg_w_pos: d.end_w_pos], axis=0)  # eq (49)

        # update distributions lambda_k over topics
        Lambda[:] = eta
        for j, w in enumerate(w for d in docs for w in d.words):  # loop through all words
            Lambda[:, w] += Phi[j]  # eq (50), for all k

    return Gamma, Lambda, Phi
