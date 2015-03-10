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

import json
from scipy import sparse


if __name__ == '__main__':
    print('Loading...')
    samples = [load_tweets('data/sample%d.txt' % i) for i in range(1,7)]
    sample1_ids = set([tw['id'] for tw in samples[0]])
    tweets = np.concatenate(samples)
    print('Loaded.')
    
    print('Cleaning...')
    tweets = remove_stop_words(tweets)
    tweets = remove_punctuation(tweets)
    tweets = remove_non_english(tweets, 0.2)
    tweets = remove_links(tweets)
    tweets = remove_digits(tweets)
    tweets = remove_empty(tweets)
    print('Cleaned.')

    #By using all tweets, we get a better idea of what the "true" use of the word is. Can detect spikes!
    print('Extracting Features...')
    feats, vocab = tf_idf(tweets, None, .995, 0.005)
    print('Features Extracted.')

    #filter out tweets not in sample1
    print('Filtering...')
    sample1_feats = feats[np.array([i for i in range(len(tweets)) if tweets[i]['id'] in sample1_ids]), :]
    print('Filtered')
    
    print('\tfull feats:', feats.shape)
    print('\tsample1_feats shape:', sample1_feats.shape)

    print('Clustering...')
    clusters = complete_linkage_clustering(sample1_feats, 120)
    print('Clustered')

##Largest cluster?
    print('Getting largest cluster...')
    largest_cluster_ind = np.argmax([sum(clusters == i) for i in range(20)])
    largest_cluster = sample1_feats[clusters==largest_cluster_ind,:]
    print('Found largest cluster.')

    #with open('data/sample1.trends.txt') as f:
    #    content = f.readlines()
    #    trends = [t['name'] for t in [json.loads(s)['trends'] for s in content] if 'name' in t]

    print('Extracting trends...')
    print(lda_extract_topic(largest_cluster, vocab, 30))
    print('Trends extracted.')
