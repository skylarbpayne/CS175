from load_tweets import load_tweets
from feature_extractors import *
from stats_and_plots import *

import matplotlib.pyplot as plt

tweets = load_tweets('data/sample1.txt')
tdm, vocab = word_counts(tweets)

print(most_used(tdm, vocab, 10))
print(least_used(tdm, vocab, 10))

plt.show(word_bar(tdm, vocab, 10))

feats = {}
feats['num_hashtags'] = num_hashtags(tweets)
feats['reply'] = reply(tweets)
feats['length'] = length(tweets)
feats['num_retweets'] = num_retweets(tweets)

print(most_hashtags(feats))
plt.show(hashtag_hist(feats))

print(proportion_replies(feats))

print(avg_tweet_length(tdm))
plt.show(length_hist(feats))

print(avg_retweets(feats))
plt.show(num_retweets_hist(feats))
