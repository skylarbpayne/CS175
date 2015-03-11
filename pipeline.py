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

if __name__ == '__main__':
    print('Loading...')
    samples = [load_tweets('data/sample%d.txt' % i) for i in range(1,7)]
    sample1_ids = set([tw['id'] for tw in samples[0]])
    tweets = np.concatenate(samples)
    print('Loaded.')
    
    print('Cleaning...')
    tweets = remove_stop_words(tweets)
    tweets = remove_punctuation(tweets)
    tweets = remove_non_english(tweets)
    tweets = remove_links(tweets)
    tweets = remove_digits(tweets)
    tweets = remove_empty(tweets)
    print('Cleaned.')

    #By using all tweets, we get a better idea of what the "true" use of the word is. Can detect spikes!
    print('Extracting Features...')
    feats, vocab = tf_idf(tweets, None, (1,3), 0.999, 0.001)
    print('Features Extracted.')

    plt.show(word_bar(feats, vocab, 50))

    #filter out tweets not in sample1
    print('Filtering...')
    sample1_feats = feats[np.array([i for i in range(len(tweets)) if tweets[i]['id'] in sample1_ids]), :]
    print('Filtered')
    
    print('\tfull feats:', feats.shape)
    print('\tsample1_feats shape:', sample1_feats.shape)

    print('Clustering...')
    n_c = 50
    clusters = ward_linkage_clustering(sample1_feats, n_c)
    print('Clustered')

##Largest cluster?
    print('Getting largest clusters...')
    cluster_sizes = [(clusters == i).sum() for i in range(n_c)]
    print(sum(cluster_sizes))
    assert(sum(cluster_sizes) == sample1_feats.shape[0])
    
    cluster_sizes = sorted([(i,sz) for i,sz in enumerate(cluster_sizes)], key=lambda x: x[1], reverse=True)
    
    expected_trends = []
    print('Found largest clusters.')
    for i,_ in cluster_sizes:
        cluster = sample1_feats[clusters==i,:]
        print('Cluster %d' % i)
        print('\tExtracting trends...')
        expected_trends.append(lda_extract_topic(cluster, vocab, 5))
        print('\tTrends extracted.')
    
    print('Expected Trends')
    print(expected_trends)
    print('Actual Trends')
    with open('data/sample1.trends.txt') as f:
        content = f.readlines()
    print([t['name'] for t in json.loads(content[0])[0]['trends']])
