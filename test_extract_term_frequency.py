from load_tweets import load_tweets
from remove_stop_words import remove_stop_words
from remove_punctuation import remove_punctuation
from remove_non_english import remove_non_english
from remove_digits import remove_digits
from remove_links import remove_links
from remove_empty import remove_empty
from feature_extractors import *
from clustering import *
from trend_extraction import *
from stats_and_plots import *

import json
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
import re

'''
A side thing that I was working on with the idea that we want to
look for spikes in term frequency over time.  Still sort of buggy.

-James
'''

def trend_extract_term_frequency(samples : list):

    # Evaluate the term frequency in each sample
    dict = {}

    for sample in samples:
        for key, value in get_word_counts(sample):
            if key in dict:
                dict[key].append(value)
            else:
                dict[key] = [value]

    li = []
    
    for key,value in dict.items():
        maxValue = max(value)
        averageValue = np.mean([t for t in value if t != maxValue])
        delta = maxValue - averageValue
        li.append((key, delta))
        print(delta)
        
    sorted_list = sorted(li, key=lambda x: x[1],reverse=True)
    
    print(sorted_list[0:5][1])
    
def get_word_counts(tweets : list) -> list:
    '''
    Takes in a list of tweets and returns a dictionary with word and its count

    :param tweets: list of tweets, where each tweet is a dictionary

    :return: dictionary containing the word and its count
    '''
    dict = {}
    cv = CountVectorizer()
    text = [t['text'] for t in tweets]
    tdm = cv.fit_transform(text)
    word_counts = tdm.sum(axis=0).tolist()[0]

    assert(len(cv.get_feature_names()) == len(word_counts))
    return zip(cv.get_feature_names(), word_counts)

if __name__ == '__main__':

    
    print('Loading...')
    samples = [load_tweets('data/sample%d.txt' % i) for i in range(1,4)]
    print('Loaded.')
    
    print('Cleaning...')
    for tweets in samples:
        tweets = remove_stop_words(tweets)
        tweets = remove_punctuation(tweets)
        tweets = remove_non_english(tweets)
        tweets = remove_links(tweets)
        tweets = remove_digits(tweets)
        tweets = remove_empty(tweets)  
    print('Cleaned.')
    

    
    print('Extract Trends')
    trend_extract_term_frequency(samples)
    print('Extracted')    


