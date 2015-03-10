import lda
import numpy as np

def lda_extract_topic(tweets: np.ndarray, vocab: list, k=1) -> str:
    '''
    :param tweets: the (sparse) document term matrix of tweets
    :param vocab: an (ordered) list of the vocabulary of the document term matrix

    :return: a string representing the trend found via LDA
    '''
    #vary these parameters
    model = lda.LDA(n_topics = k, n_iter = 500)
    model.fit(tweets)

    return np.array(vocab)[np.argsort(model.topic_word_[0])][-1]
