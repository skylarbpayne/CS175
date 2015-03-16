import lda
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from stats_and_plots import *

def lda_extract_topic(tweets: np.ndarray, vocab: list, k=1) -> str:
    '''
    :param tweets: the (sparse) document term matrix of tweets
    :param vocab: an (ordered) list of the vocabulary of the document term matrix

    :return: a string representing the trend found via LDA
    '''
    #vary these parameters
    model = lda.LDA(n_topics = k, n_iter = 1000)
    model.fit(tweets)

    return np.array(vocab)[np.argsort(model.topic_word_[0:4])][-1]

def tf_extract_topic(tweets: np.ndarray, vocab: list, k=1) -> str:
    '''
    This is copy/pasta of most_used.  Most stable version that atleast returns the 'most used term'
    '''
    word_counts = sorted(enumerate(tweets.sum(axis=0).tolist()[0]), key=lambda s: s[1])[::-1]
    return [(vocab[i],count) for i,count in word_counts][0][0]
