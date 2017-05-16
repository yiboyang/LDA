"""
Test lda + linear regression on Amazon review data
"""

if __name__ == '__main__':  # for parallel processing on Windows
    import os
    import pickle

    import sys

    sys.path.append(os.pardir)  # hack for running from command line

    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn import linear_model

    from collections import Counter

    from LDA import utils
    from LDA import lda

    pkl_name = 'software_1000.pkl'
    data_dir = os.path.join(os.pardir, 'data')
    pkl_path = os.path.join(data_dir, pkl_name)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    ints_list = data['text_ints_list']
    responses = np.array(data['responses'])
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

    train_responses = np.log(np.array(train_responses))

    test_corpus = ints_list[T:]
    test_docs = utils.make_docs(test_corpus)
    test_responses = responses[T:]
    print('test labels:', Counter(test_responses))

    np.random.seed(0)
    K = 10
    alpha = 1 / K

    # babysitting VEM training
    niter = 10
    tol, max_try = 0.1, 5
    record = {'mse': [], 'pr2': [], 'accuracy': [], 'close': [], 'conf_mat': [],
              'Beta': [], 'eta': []}

    # copied from LDA.py vem
    Beta = np.random.dirichlet(np.ones(V), K)  # model params

    # variational params
    Gamma = np.random.rand(len(train_docs), K) + alpha + np.array([d.len / K for d in train_docs])[:, np.newaxis]
    Phi = np.random.dirichlet(alpha=np.ones(K) * alpha, size=sum(d.len for d in train_docs))

    for it in range(niter):
        Gamma, Phi = lda.vem_estep(train_docs, K, alpha, Beta, prev_params=(Gamma, Phi), tol=0.1, max_try=5)
        Beta = lda.vem_mstep(train_docs, K, V, Phi)  # optimized model parameters

        # check on the topics learned
        n = 10  # top n terms
        for lamb in Beta:
            idx = np.argpartition(lamb, -n)[-n:]
            idx = reversed(idx[np.argsort(lamb[idx])])
            print([int2word[i] for i in idx])
            print()

        X_train = (Gamma - alpha) / np.array([d.len for d in train_docs])[:, np.newaxis]
        regr = linear_model.LinearRegression()
        regr.fit(X_train, train_responses)
        eta = regr.coef_

        # test after every VEM iteration
        print('test')
        test_Gamma, test_Phi = lda.vem_estep(test_docs, K, alpha, Beta, tol=tol*.1,
                                             max_try=max_try*2)

        X_test = (test_Gamma - alpha) / np.array([d.len for d in test_docs])[:, np.newaxis]
        test_predictions = regr.predict(X_test)
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

        for k, v in sorted(record.items()):
            if k in {'mse', 'pr2', 'accuracy', 'close', 'conf_mat'}:
                print('%s: %s' % (k, v[-1]))
            if k == 'eta':
                print('eta:', eta - np.mean(eta))  # more interpretable

    # save_name = 'N%d_K%d_T%d' % (N, K, niter)
    # print('saving as ', save_name)
    # with open(save_name + '.pkl', 'wb') as f:
    #     pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)

    plt.figure()
    for k, v in sorted(record.items()):
        if k in {'mse', 'pr2', 'accuracy', 'close'}:
            plt.plot(v, label=k)

    plt.legend(loc='best')
    plt.show()
    # plt.savefig(save_name)
