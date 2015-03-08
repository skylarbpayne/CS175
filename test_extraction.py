from trend_extraction import lda_extract_topic
from feature_extractors import word_counts
from load_tweets import load_tweets

if __name__ == '__main__':
    tweets = load_tweets('data/sample1.txt')
    feats, vocab = word_counts(tweets)
    print(lda_extract_topic(feats, vocab))
