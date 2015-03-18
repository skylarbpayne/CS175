from load_tweets import load_tweets
from remove_stop_words import remove_stop_words
from remove_punctuation import remove_punctuation
from remove_non_english import remove_non_english
from remove_digits import remove_digits
from remove_links import remove_links
from remove_empty import remove_empty
from remove_small_tokens import remove_small_tokens
from feature_extractors import *
from clustering import *
from trend_extraction import *
from stats_and_plots import *

import json
from scipy import sparse
import matplotlib.pyplot as plt

'''
Short outline:
    Construct a big dataset of all those pieces
    clean data
    fit tdm to data
    transform entire set of tweets
    cluster (only set that I want to know about)
    big clusters -> take the sum over rows / # tweets in cluster
    subtract from sum over rows / # tweets total.
    take max over that difference
    find corresponding word.
'''

if __name__ == '__main__':
    print('Loading...')
    samples = [load_tweets('data/sample%d.txt' % i) for i in range(1,7)]
    sample1_ids = set([tw['id'] for tw in samples[0]])
    #we might want to package up ALL tweets -- we'll try that if this doesn't work.
    tweets = np.concatenate(samples) #package up all tweets except ones I'm currently analyzing.
    print('Loaded.')
    
    print('Cleaning...')
    tweets = remove_punctuation(tweets)
    tweets = remove_non_english(tweets, 0.35)
    tweets = remove_small_tokens(tweets, 3)
    tweets = remove_stop_words(tweets)
    tweets = remove_links(tweets)
    tweets = remove_digits(tweets)
    tweets = remove_empty(tweets)
    print('Cleaned.')

    #By using all tweets, we get a better idea of what the "true" use of the word is. Can detect spikes!
    print('Extracting Features...')
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(1,3), max_df=0.7, min_df=0.00025)
    cv.fit([t['text'] for t in tweets])
    feats = cv.transform([t['text'] for t in tweets])

    feats_sample1 = feats[np.array([i for i in range(len(tweets)) if tweets[i]['id'] in sample1_ids]), :]
    print('Features Extracted.')

    #plt.show(word_bar(feats, vocab, 50))
    
    print('\tfull feats:', feats.shape)
    print('\tsample1_feats shape:', feats_sample1.shape)

    print('Clustering...')
    n_c = 12
    clusters = k_means_clustering(feats_sample1, n_c)
    print('Clustered')

##Largest cluster?
    print('Getting largest clusters...')
    cluster_sizes = [(clusters == i).sum() for i in range(n_c)]
    
    cluster_sizes = sorted([(i,sz) for i,sz in enumerate(cluster_sizes)], key=lambda x: x[1], reverse=True)
    avg_term_usage = feats.sum(axis=0) / feats.shape[0]
    
    expected_trends = []
    print('Found largest clusters.')
    for i,_ in cluster_sizes:
        cluster = feats_sample1[clusters==i,:]
        print('Cluster %d' % i)
        print('\tExtracting trends...')
        avg_cluster_term_usage = cluster.sum(axis=0) / cluster.shape[0]
        dif = (avg_cluster_term_usage - avg_term_usage).reshape((avg_term_usage.shape[1],1))
        possible_trends = sorted([(i,diff) for i,diff in enumerate(dif)], key=lambda x: x[1], reverse=True)
        expected_trends.append([cv.get_feature_names()[i] for i,_ in possible_trends[0:5]])

        #expected_trends.append(lda_extract_topic(cluster, vocab, 5))
        print('\tTrends extracted.')
    
    print('Expected Trends')
    print(expected_trends)
    print('Actual Trends')
    with open('data/sample1.trends.txt') as f:
        content = f.readlines()
    print([t['name'] for t in json.loads(content[0])[0]['trends']])

    print('\nWords:')
    print(cv.get_feature_names()[1:])
