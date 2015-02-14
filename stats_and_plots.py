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
    word_counts = sorted(enumerate(np.sum(tdm, axis=0)), key=lambda s: s[1])[::-1]
    return [(feat_names[i],count) for i,count in word_counts][:k]

def least_used(tdm: np.ndarray, k: int) -> list:
    word_counts = sorted(enumerate(np.sum(tdm, axis=0)), key=lambda s: s[1])
    return [(feat_names[i],count) for i,count in word_counts][:k]

def word_hist(tdm: np.ndarray, bins: int = None) -> plt.Figure:
    return plt.hist(np.sum(tdm, axis=0), bins)

def most_hashtags(feats: np.ndarray) -> int:
    return feats['num_hashtags'].max()

def hashtag_hist(feats: np.ndarray, bins: int = None) -> plt.Figure:
    return plt.hist(feats['num_hashtags'], bins)

def proportion_replies(feats: np.ndarray) -> float:
    return feats['reply'].sum() / len(feats['reply'])

def avg_tweet_length(tdm: np.ndarray) -> float:
    return np.sum(tdm) / np.shape(tdm)[0]


