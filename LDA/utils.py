import itertools

import nltk
import numpy as np

from .lda import Document


def counter_to_vec(counter, length=None, dtype=int):
    """
    Convert a Counter object constructed over non-negative integers to a vector of counts.
    >>> counter_to_vec(Counter([0, 1, 1, 2, 1, 2]), length=4)
    numpy.array([1, 3, 2, 0])

    :param counter: Counter of integer values >= 0
    :param length: length of the resulting vector; must be greater than the largest key
    :param dtype:
    :return: a vector of counts of integers, indexed by [0, 1, ..., length-1]
    """
    if len(counter) == 0:  # special case of empty counter
        return np.zeros(length, dtype=dtype)

    max_key = max(counter.keys())
    if length is None:
        length = max_key + 1
    else:
        assert length > max_key, "length less than or equal to max key in counter!"

    keys, vals = zip(*counter.items())
    vec = np.zeros(length, dtype=dtype)
    vec[list(keys)] = vals
    return vec


def make_docs(ints_list):
    """
    list of integer lists => list of Documents
    :param ints_list:
    :return:
    """
    docs = []
    w_pos = 0  # keep track of word position in corpus
    for i, text in enumerate(ints_list):
        d = Document(words=text, beg_w_pos=w_pos)
        docs.append(d)
        w_pos += d.len
    return docs


def get_stop_words():
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    return stop


def get_punctuations():
    import string
    punctuations = set(string.punctuation)
    punctuations |= {'>,', '>.', ',"', '."', ').', '),', '--', '..', '...', "''", '``'}
    return punctuations


def pre_process_text(words, remove_stop=True, remove_punc=True, remove_num=False, min_len=2, lower_case=True,
                     stem=False):
    """
    Convenience method for pre-processing a single piece of text.
    >>> pre_process_text("I have 3 cats.", remove_num=True, stem='porter')
    ... ["i", "have", "cat"]
    :param words: a text string, or a list of words (already tokenized)
    :param remove_stop:
    :param remove_punc:
    :param remove_num:
    :param min_len: minimum length of word to be retained
    :param lower_case:
    :param stem: perform stemming use specified stemmer; 'porter', 'snowball', or 'lancaster'
    :return: a list of pre-processed words
    """
    stop_words = get_stop_words()
    puncts = get_punctuations()
    if stem:
        # assert isinstance(stem, str)
        stem = stem.lower()
        if stem == 'porter':
            stemmer = nltk.stem.PorterStemmer()
        elif stem == 'snowball':
            stemmer = nltk.stem.SnowballStemmer(language='english')  # note Snowball will automatically lowercase
        elif stem == 'lancaster':
            stemmer = nltk.stem.LancasterStemmer()
        else:
            raise ValueError('invalid stemmer name')

    if isinstance(words, str):
        raw_words = nltk.word_tokenize(words)
    else:
        # assert isinstance(words, list)
        raw_words = words
    valid_words = []
    for w in raw_words:
        if remove_stop and w in stop_words:
            continue
        if remove_punc and w in puncts:
            continue
        if remove_num and w.isnumerc():
            continue
        if min_len and len(w) <= min_len:
            continue
        if lower_case:
            w = w.lower()
        if stem:
            w = stemmer.stem(w)

        valid_words.append(w)
    return valid_words


def remove_outlier_terms(words_list, min_idc=5, max_idc=None, min_idf=0.01, max_idf=0.25):
    """
    Remove terms that are too common or too infrequent. Goal is to retain words with high discriminative power.
    :param words_list: a list of list of words, e.g. [['hi', 'there'], ['how','are'],...]
    :param min_idc: remove if appears in less than this number of documents (word lists)
    :param max_idc: remove if appears in more than this number of documents (word lists)
    :param min_idf: remove if appears in less than this fraction of documents (word lists)
    :param max_idf: remove if appears in more than this fraction of documents (word lists)
    :return: a copy of words_list with outliers removed
    """
    from collections import Counter
    outliers = set()  # set of terms ot be removed

    N = len(words_list)  # total num of "documents"
    terms_list = [set(words) for words in words_list]  # list of unique terms in each word list
    vocab = set(itertools.chain(*terms_list))
    idc = Counter()  # inverse document count

    for terms in terms_list:
        terms_count = Counter(terms)
        idc += terms_count

    # remove by inverse document count
    for term in vocab:
        c = idc[term]  # how many documents this term appears in
        if max_idc and c > max_idc:  # if term appears in more than max_idc # of documents
            outliers.add(term)
        if min_idc and c < min_idc:
            outliers.add(term)
        idf = c / N  # inverse document frequency; what fraction of documents the term appears in
        if min_idf and idf < min_idf:
            outliers.add(term)
        if max_idf and idf > max_idf:
            outliers.add(term)

    # remove outliers
    new_words_list = []
    for words in words_list:
        new_words_list.append([w for w in words if w not in outliers])
    return new_words_list


def words_list_to_ints_list(words_list, word2int):
    """
    Convenience method for converting lists of lists of words to lists of lists of integers.
    >>> words_list_to_ints_lists([['hi', 'there'], ['how','are']], {'hi':0,'how':1,'there':2,'are':3})
    ... [[0, 2], [1, 3]]
    :param words_list:
    :param word2int: dict
    :return: list of integer lists
    """
    ints_list = []
    for words in words_list:
        ints_list.append([word2int[w] for w in words])
    return ints_list


def map_word_int(words_list):
    """
    Map each word in corpus to an integer.
    :param words_list: a list of list of words, e.g. [['hi', 'there'], ['how','are'],...]
    :return two dicts: word2int, int2word
    """
    terms = set(itertools.chain(*words_list))  # vocab
    V = len(terms)
    word2int = {term: index for term, index in zip(terms, range(V))}
    int2word = {v: k for (k, v) in word2int.items()}  # reverse mapping
    return word2int, int2word
