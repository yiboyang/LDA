"""
supervised LDA

Based on the sLDA paper http://www.cs.columbia.edu/~blei/papers/BleiMcAuliffe2007.pdf
For more details see https://arxiv.org/pdf/1003.0783.pdf
'eta' denote the vector of weights in the exp linear component of GLM
'var' means the dispersion parameter of GLM; sigma squared for Gaussian
Here we use a normal linear model for regression.
"""

import numpy as np
import scipy.optimize
import scipy.stats
from scipy.special import gammaln, digamma


def calc_E_ATA_per_doc(d_len, Phi_d, Phi_sum_d):
    """
    Helper function for computing expected sufficient stats;
    equivalent to equation (5) in sLDA paper but more efficient
    :param d_len:
    :param Phi_d: Phi[d.beg_w_pos:d.end_w_pos]
    :param Phi_sum_d: Phi[d.beg_w_pos:d.end_w_pos].sum(axis=0)
    :return:
    """
    all_outer_prod_sum = np.einsum('i,kj->ij', Phi_sum_d, Phi_d)
    # above is equivalent to:
    # np.sum([np.outer(Phi[n], Phi[m]) for n in range(len(Phi)) for m in range(len(Phi))], axis=0)
    return (all_outer_prod_sum - np.dot(Phi_d.T, Phi_d) + np.diag(Phi_sum_d)) / (d_len) ** 2


def expected_moments(docs, Phi, njobs=1):
    """
    Expected sufficient statistics (first/second) moments under variational posterior distribution
    :param docs:
    :param Phi:
    :param njobs: number of processes to launch for parallel processing
    :return:
    """
    Phi_sums = np.array([np.sum(Phi[d.beg_w_pos:d.end_w_pos], axis=0) for d in docs])  # DxK; not worth paralleling
    E_A = Phi_sums / np.array([d.len for d in docs])[:, np.newaxis]

    if njobs > 1:
        from joblib import Parallel, delayed
        parallelizer = Parallel(n_jobs=njobs)
        tasks_iterator = (delayed(calc_E_ATA_per_doc)(d.len, Phi[d.beg_w_pos:d.end_w_pos], Phi_sums[i]) for i, d in
                          enumerate(docs))
        partial_results = parallelizer(tasks_iterator)
        E_ATA = np.sum(partial_results, axis=0)
    else:  # vanilla for loop; faster for smaller corpus (< 1M) because of parallelization overhead
        K = Phi.shape[1]
        E_ATA = np.zeros((K, K))
        for i, d in enumerate(docs):
            E_ATA += calc_E_ATA_per_doc(d.len, Phi[d.beg_w_pos:d.end_w_pos], Phi_sums[i])

    return E_A, E_ATA


def vem_estep(docs, resps, K, alpha, Beta, eta, var, stats, prev_params=None, tol=0.01, max_try=20):
    """
    Variational Bayesian inference for sLDA. We adopt a semi-Bayesian approach and maintain independent variational
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
    :param eta: GLM weight param
    :param var: GLM variance param
    :param prev_params: variational parameters from previous iteration, (Gamma, Phi); if provided will generally
    make vem more efficient
    :param tol: float, error tolerance, the minimum percentage increase of elbo, below which the algorithm is
    considered to have converged
    :param max_try: int; if elbo has not increased by more than tol for max_try iterations, return
    :return:
    """

    def elbo(E_log_Theta, log_Beta, Gamma, Phi):
        """
        Calculate the variational lower bound on log evidence.
        We use the equation L = E_q[log p(X,Z)] + H[log q(Z)], where H is the entropy; for LDA it can be decomposed
        L = E_q[log p(theta|alpha)] + E_q[log p(z,w|theta,beta)] + E_q[log p(y|z,eta,sigma)] +
        H[q(theta|gamma)] + H[q(z|phi)]
        The unspecified parameters are constant global variables.
        :param E_log_Theta: expectation of log theta wrt q(theta|Gamma), calculated in e step for efficiency.
        :param log_Beta:
        :param Gamma:
        :param Phi:
        :return: variational lower bound for naive mean field
        """
        E_log_p_theta = (alpha - 1) * E_log_Theta.sum() + D * (gammaln(alpha * K) -
                                                               K * gammaln(alpha))  # line 1 of eq (15)

        E_log_p_zw = 0
        for i, d in enumerate(docs):
            E_log_p_zw += np.sum((E_log_Theta[i] + log_Beta[:, d.words].T) *
                                 Phi[d.beg_w_pos: d.end_w_pos])  # line 2,3 of eq (15) combined

        E_log_p_y = -0.5 * np.log(2 * np.pi * var) - 1 / (2 * var) * \
                                                     (resps_norm_sq - 2 * np.dot(resps, np.dot(E_A, eta)) +
                                                      np.dot(np.dot(eta, E_ATA), eta))  # sLDA eq (9)

        H_q_theta = sum(scipy.stats.dirichlet.entropy(g) for g in Gamma)

        H_q_z = -(Phi * np.log(Phi)).sum()

        lb = E_log_p_theta + E_log_p_zw + E_log_p_y + H_q_theta + H_q_z
        return lb

    D = len(docs)
    W = sum(d.len for d in docs)  # total # of words in corpus
    eta_prod = eta * eta  # Hadamard product
    resps_norm_sq = np.dot(resps, resps)  # resps L2 norm squared
    E_A, E_ATA = stats

    if prev_params is None:
        # random initialization based on Blei's paper figure 6
        Gamma = np.random.rand(D, K) + alpha + np.array([d.len / K for d in docs])[:, np.newaxis]
        Phi = np.random.dirichlet(alpha=np.ones(K) * alpha, size=W)
    else:
        Gamma, Phi = prev_params

    lb_prev = float("inf")  # ELBO from previous iteration

    while True:
        # do some preliminary calculations for lower bound computation as well as parameter updates;
        # try to operate on large matrices for efficiency on PC; may take forever :)

        E_log_Theta = digamma(Gamma) - digamma(Gamma.sum(axis=1))[:, np.newaxis]  # eq (8), for all thetas
        log_Beta = np.log(Beta + 1e-32)  # for numeric stability

        lb = elbo(E_log_Theta, log_Beta, Gamma, Phi)
        print(lb)
        if lb_prev != float("inf") and abs((lb - lb_prev) / lb_prev) < tol:  # if no improvement in elbo
            if num_try > 0:
                num_try -= 1
            else:  # num_try == 0
                break
        else:  # if there was improvement in elbo
            num_try = max_try
            lb_prev = lb

        for i, d in enumerate(docs):  # unfortunately un-parallelizable, has to be done sequentially
            Phi_sum = np.sum(Phi[d.beg_w_pos: d.end_w_pos], axis=0)
            for j, w in zip(range(d.beg_w_pos, d.end_w_pos), d.words):
                y = resps[i]
                Phi_sum -= Phi[j]
                log_Phi_new_j = E_log_Theta[i] + log_Beta[:, w] + (y / d.len / var) * eta - \
                                (2 * np.dot(eta, Phi_sum) * eta + eta_prod) / (
                                    2 * (d.len) ** 2 * var)  # eq (7) of sLDA paper
                Phi_new_j = np.exp(log_Phi_new_j)
                Phi[j] = Phi_new_j / np.sum(Phi_new_j)
                Phi_sum += Phi[j]

            Gamma[i] = alpha + Phi_sum  # update distributions gamma_d over document; re-use Phi_sum calculation

    return Gamma, Phi


def vem_mstep(docs, resps, K, V, Phi, stats):
    """
    M-step of variational EM to estimate hyperparameters using maximum likelihood based on expected sufficient stats
    under approximate posterior; same as maximizing ELBO wrt to Beta, eta, and var
    :return:
    """
    D = len(docs)
    Beta = np.zeros((K, V))  # categorical param for topics
    for j, w in enumerate(w for d in docs for w in d.words):  # loop through all words
        Beta[:, w] += Phi[j]
    Beta /= np.sum(Beta, axis=1)[:, np.newaxis]  # eq (9), MLE for Beta

    # MLE for the GLM params
    E_A, E_ATA = stats
    E_AT_y = np.dot(E_A.T, resps)
    eta = np.linalg.solve(E_ATA, E_AT_y)
    var = (1 / D) * (np.dot(resps, resps) - np.dot(E_AT_y.T, eta))

    print('MSE:', np.sum((np.dot(E_A, eta) - resps) ** 2) / D)  # should decrease
    return Beta, eta, var


def vem(docs, resps, alpha, K, V, niter, njobs=1):
    # model params
    Beta = np.random.dirichlet(np.ones(V), K)
    eta = np.linspace(start=-1, stop=1, num=K)  # initialization based on bottom of page 7 of sLDA paper
    var = np.var(resps)

    # variational params
    Gamma = np.random.rand(len(docs), K) + alpha + np.array([d.len / K for d in docs])[:, np.newaxis]
    Phi = np.random.dirichlet(alpha=np.ones(K) * alpha, size=sum(d.len for d in docs))

    for it in range(niter):
        stats = expected_moments(docs, Phi, njobs)
        Beta, eta, var = vem_mstep(docs, resps, K, V, Phi, stats)
        Gamma, Phi = vem_estep(docs, resps, K, alpha, Beta, eta, var, stats,
                               prev_params=(Gamma, Phi), tol=0.1, max_try=5)

    return Beta, eta, var  # optimized model parameters


def predict(docs, alpha, K, Beta, eta, tol=0.001, max_try=20):
    """
    Label a new set of documents with responses
    :param docs:
    :param alpha:
    :param K:
    :param V:
    :param Beta:
    :param eta:
    :return:
    """
    # run inference in the original LDA model to obtain the covariates (phi bars for each doc)
    from . import lda

    _, Phi = lda.vem_estep(docs=docs, K=K, alpha=alpha, Beta=Beta, prev_params=None, tol=tol, max_try=max_try)
    E_A = np.empty((len(docs), K))
    for i, d in enumerate(docs):
        E_A[i] = np.sum(Phi[d.beg_w_pos:d.end_w_pos], axis=0) / d.len
    return np.dot(E_A, eta)
