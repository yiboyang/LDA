import os
import pickle
import random

import sys
sys.path.append(os.pardir)

import LDA.utils as utils


def yield_json_from_gz(gz_path):
    import gzip
    g = gzip.open(gz_path, 'r')
    for l in g:
        yield eval(l)  # each item is a json dict


def get_json(json_path):
    import json
    with open(json_path) as f:
        return json.load(f)


dataset_name = 'software'
dataset_path = 'reviews_Software_5.json.gz'

num_to_extract = 5000
num_to_extract_per_class = [num_to_extract / 5] * 5  # there're five rating classes in total; want the same num each
min_text_length = 10  # may want to set reasonably high so we don't get rid of docs due to pre-processing

words_list = []
responses = []

# dataset specific code
for d in yield_json_from_gz(dataset_path):
    text = d['summary'] + ' ' + d['reviewText']  # use both
    words = utils.pre_process_text(text, remove_stop=True, remove_punc=True, remove_num=False, min_len=2,
                                   lower_case=True, stem=None)  # word level preprocessing
    if len(words) < min_text_length:  # simple pre-screening
        continue
    else:
        response = int(d['overall'])  # int in [1,2,3,4,5]
        response_class = int(response - 1)
        if num_to_extract_per_class[response_class] > 0:  # if still need more from this class
            words_list.append(words)
            responses.append(response)
            num_to_extract_per_class[response_class] -= 1
        else:
            if sum(num_to_extract_per_class) == 0:  # if extracted all we need for each class
                break
            else:
                continue

print('corpus level preprocessing')  # this is where we might get rid of less discriminative documents
words_list = utils.remove_outlier_terms(words_list, min_idc=5, max_idc=None, min_idf=None, max_idf=0.25)
# make sure to get rid of empty documents and responses as a result of possibly aggressive pre-processing
wr = [(w, r) for w, r in zip(words_list, responses) if len(w) > 0]
assert len(wr) == num_to_extract, "got %d after preprocessing, less than %d!" % (len(wr), num_to_extract)
random.shuffle(wr)  # shuffle to get roughly uniform distribution of reviews
w, r = zip(*wr)
words_list = list(w)
responses = list(r)
num_docs = len(words_list)

# done pre-processing; convert to integers
word2int, int2word = utils.map_word_int(words_list)
ints_list = utils.words_list_to_ints_list(words_list, word2int)  # list of integer arrays

# build a data dictionary to be saved
data = {'text_words_list': words_list, 'text_ints_list': ints_list, 'responses': responses,
        'word2int': word2int, 'int2word': int2word, 'name': dataset_name}

pkl_path = os.path.join(os.curdir, '%s_%d.pkl' % (dataset_name, num_docs))
try:
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('wrote to ' + pkl_path)
except Exception as e:
    print('Unable to save data to', pkl_path, ':', e)
