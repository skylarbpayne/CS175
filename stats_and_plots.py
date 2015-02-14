import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


'''
Stats and plots:
    -10 most used words
    -10 least used words
    -Histogram of word usage
    -Greatest number of hashtags used
    -Histogram of number of hashtags used
    -Proportion of tweets that are replies
    -Average tweet length

Produce these stats/plots for tweets with and without stopwords removed and also with
various thresholds.
'''

def most_used(tdm: np.ndarray, feat_names: list, k: int) -> list:
    '''
    Returns the most used tokens according to a term document matrix

    :param tdm: The term document matrix
    :param feat_names: the vocabulary list
    :param k: the number of most used tokens to return

    :return: a list of the k most used words
    '''

    word_counts = sorted(enumerate(np.sum(tdm, axis=0)), key=lambda s: s[1])[::-1]
    return [(feat_names[i],count) for i,count in word_counts][:k]

def least_used(tdm: np.ndarray, k: int) -> list:
    '''
    Returns the least used tokens according to a term document matrix
    
    :param tdm: The term document matrix
    :param feat_names: the vocabulary list
    :param k: the number of least used tokens to return

    :return: a list of the k least used words
    '''

    word_counts = sorted(enumerate(np.sum(tdm, axis=0)), key=lambda s: s[1])
    return [(feat_names[i],count) for i,count in word_counts][:k]

def word_hist(tdm: np.ndarray) -> plt.Figure:
    '''
    Plots a histogram of the use of all words

    :param tdm: the term document matrix
    
    :return: the figure of the histogram plot
    '''

    return plt.hist(np.sum(tdm, axis=0))

def most_hashtags(feats: np.ndarray) -> int:
    '''
    Returns the greatest frequency of hashtags

    :param feats: the features of the tweets (must include 'num_hashtags')
    
    :return: the greatest number of used hashtags in a single tweet
    '''

    return feats['num_hashtags'].max()

def hashtag_hist(feats: np.ndarray) -> plt.Figure:
    '''
    A histogram of the frequency of hashtags in individual tweets

    :param feats: the features of the tweets ( must include 'num_hashtags')

    :return: the figure of the histogram plot
    '''

    return plt.hist(feats['num_hashtags'], bins)

def proportion_replies(feats: np.ndarray) -> float:
    '''
    Returns the proportion of tweets that are a reply vs all tweets

    :param feats: the features of the tweets (must include 'reply')

    :return: the proportion of tweets that are reply vs all tweets
    '''

    return feats['reply'].sum() / len(feats['reply'])

def avg_tweet_length(tdm: np.ndarray) -> float:
    '''
    Returns the average length of a tweet

    :param tdm: the term document matrix

    :return: the average length of a tweet
    '''

    return np.sum(tdm) / np.shape(tdm)[0]
