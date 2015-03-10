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

if __name__ == '__main__':
    tweets = load_tweets('data/sample1.txt')
    tweets = remove_stop_words(tweets)
    tweets = remove_punctuation(tweets)
    tweets = remove_non_english(tweets, 0.2)
    tweets = remove_links(tweets)
    tweets = remove_digits(tweets)
    tweets = remove_empty(tweets)
    feats, vocab = tf_idf(tweets, None, 0.994, 0.006)

    print(feats.shape)
    clusters = complete_linkage_clustering(feats, 120)
    
##Largest cluster?
    largest_cluster_ind = np.argmax([sum(clusters == i) for i in range(20)])
    largest_cluster = feats[clusters==largest_cluster_ind,:]
    
    #with open('data/sample1.trends.txt') as f:
    #    content = f.readlines()
    #    trends = [t['name'] for t in [json.loads(s)['trends'] for s in content] if 'name' in t]


    print(lda_extract_topic(largest_cluster, vocab, 30))
