from load_tweets import load_tweets
from remove_stop_words import remove_stop_words
from remove_punctuation import remove_punctuation
from remove_non_english import remove_non_english
from remove_links import remove_links
from feature_extractors import *
from clustering import *
from trend_extraction import *

if __name__ == '__main__':
    tweets = load_tweets('data/sample1.txt')
    tweets = remove_stop_words(tweets)
    tweets = remove_punctuation(tweets)
    tweets = remove_non_english(tweets, 0.5)
    tweets = remove_links(tweets)

    feats, vocab = word_counts(tweets)

    clusters = complete_linkage_clustering(feats, 50)
    
##Largest cluster?
    largest_cluster_ind = np.argmax([sum(clusters == i) for i in range(20)])
    largest_cluster = feats[clusters==largest_cluster_ind,:]
    print(lda_extract_topic(largest_cluster, vocab))
