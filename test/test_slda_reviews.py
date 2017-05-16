"""
Test slda on Amazon review data
"""

if __name__ == '__main__':  # for parallel processing on Windows
    import os
    import pickle

    import sys

    sys.path.append(os.pardir)  # hack for running from command line

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    from collections import Counter

    from LDA import utils
    from LDA import slda

    pkl_name = 'software_1000.pkl'
    data_dir = os.path.join(os.pardir, 'data')
    pkl_path = os.path.join(data_dir, pkl_name)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    ints_list = data['text_ints_list']
    responses = data['responses']
    word2int = data['word2int']
    int2word = data['int2word']

    N = len(responses)  # total number of instances
    V = len(word2int)  # vocab size
    print('V =', V)

    T = int(0.8 * N)  # training num docs
    train_corpus = ints_list[:T]
    train_docs = utils.make_docs(train_corpus)
    print('W =', sum(d.len for d in train_docs))  # total # of words in corpus
    train_responses = responses[:T]
    print('train labels:', Counter(train_responses))

    # massage the responses to have roughly Gaussian distribution
    # train_responses_mean = np.mean(train_responses)
    # train_response_std = np.std(train_responses)
    # train_responses = (train_responses - train_responses_mean) / train_response_std
    train_responses = np.log(np.array(train_responses))

    test_corpus = ints_list[T:]
    test_docs = utils.make_docs(test_corpus)
    test_responses = responses[T:]
    print('test labels:', Counter(test_responses))

    np.random.seed(0)
    K = 5
    alpha = 1 / K

    # # unsupervised LDA just to get a feel
    # Gamma, Lambda, Phi = LDA.vi(docs=train_docs, K=K, V=V, alpha=alpha, eta=0.1, tol=0.001)
    # # get an idea of the most common words under each topic
    # n = 10  # top n
    # for lamb in Lambda:
    #     idx = np.argpartition(lamb, -n)[-n:]
    #     idx = reversed(idx[np.argsort(lamb[idx])])
    #     print([int2word[i] for i in idx])
    #     print()

    # babysitting VEM training
    niter = 10
    tol, max_try = 0.1, 5
    docs, resps = train_docs, train_responses
    record = {'mse': [], 'pr2': [], 'accuracy': [], 'close': [], 'conf_mat': [],
              'Beta': [], 'eta': [], 'var': []}

    # copied from sLDA.py vem
    Beta = np.random.dirichlet(np.ones(V), K)
    eta = np.linspace(start=-1, stop=1, num=K)  # initialization based on bottom of page 7 of sLDA paper
    var = np.var(resps)

    # variational params
    Gamma = np.random.rand(len(docs), K) + alpha + np.array([d.len / K for d in docs])[:, np.newaxis]
    Phi = np.random.dirichlet(alpha=np.ones(K) * alpha, size=sum(d.len for d in docs))

    for it in range(niter):
        stats = slda.expected_moments(docs, Phi, njobs=1)
        Beta, eta, var = slda.vem_mstep(docs, resps, K, V, Phi, stats)
        Gamma, Phi = slda.vem_estep(docs, resps, K, alpha, Beta, eta, var, stats,
                                    prev_params=(Gamma, Phi), tol=tol, max_try=max_try)

        # check on the topics learned
        n = 10  # top n terms
        for lamb in Beta:
            idx = np.argpartition(lamb, -n)[-n:]
            idx = reversed(idx[np.argsort(lamb[idx])])
            print([int2word[i] for i in idx])
            print()

        # test after every VEM iteration
        print('test')
        test_predictions = slda.predict(test_docs, alpha, K, Beta, eta, tol=tol * .1,
                                        max_try=max_try * 2)  # want more accurate inference
        test_predictions = np.exp(test_predictions)

        sum_squared_err = np.sum((test_responses - test_predictions) ** 2)
        mse = sum_squared_err / (N - T)  # mean squared error

        pr2 = 1 - sum_squared_err / np.sum((test_responses - np.mean(test_responses)) ** 2)  # predictive R^2

        test_classes = np.rint(test_predictions)  # round to get integer classification
        accuracy = np.sum(test_classes == test_responses) / (N - T)

        close = np.sum(np.abs(test_classes - test_responses) < 2) / (N - T)  # +-1 within correct label

        conf_mat = confusion_matrix(test_responses, test_classes)

        record['mse'].append(mse)
        record['pr2'].append(pr2)
        record['accuracy'].append(accuracy)
        record['close'].append(close)
        record['conf_mat'].append(conf_mat)

        record['Beta'].append(Beta)
        record['eta'].append(eta)
        record['var'].append(var)

        for k, v in sorted(record.items()):
            if k in {'mse', 'pr2', 'accuracy', 'close', 'conf_mat', 'eta'}:
                print('%s: %s' % (k, v[-1]))

    save_name = 'N%d_K%d_T%d' % (N, K, niter)
    print('saving as ', save_name)
    with open(save_name + '.pkl', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)

    plt.figure()
    for k, v in sorted(record.items()):
        if k in {'mse', 'pr2', 'accuracy', 'close'}:
            plt.plot(v, label=k)

    plt.legend(loc='best')
    # plt.show()
    plt.savefig(save_name)
