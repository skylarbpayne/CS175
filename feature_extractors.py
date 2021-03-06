import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer import tokenize

def word_counts(tweets: list, k: int = None) -> (np.ndarray, list):
    '''
    Takes a list of dictionary objects representing tweets. Returns a sparse word count matrix of size (num tweets) by (num words)

    :param tweets: The list of tweets, where each tweet is a dictionary
    
    :return: tuple of (count matrix, vocabulary list)
    '''
    
    cv = CountVectorizer(max_features=k)
    text = [t['text'] for t in tweets]
    return cv.fit_transform(text), cv.get_feature_names()
    
def tf_idf(tweets: list, k: int = None, ngram_range=(1,1), max_df=1.0, min_df=1) -> (np.ndarray, list):
    '''
    Takes a list of dictionary objects representing tweets. Returns a tf-idf word matrix of size (num tweets) by (num words)

    :param tweets: The list of tweets, where each tweet is a dictionary
    
    :return: tuple of (tf-idf matrix, vocabulary list)
    '''
    
    tv = TfidfVectorizer(tokenizer=tokenize, max_features=k, ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    text = [t['text'] for t in tweets]
    return tv.fit_transform(text), tv.get_feature_names()

def num_hashtags(tweets: list) -> list:
    '''
    Takes a list of dictionary objects representing tweets. Returns the number of hashtag counts for each tweet

    :param tweets: The list of tweets, where each tweet is a dictionary
    
    :return: list of hashtag counts
    '''

    return [t['text'].count('#') for t in tweets]

def reply(tweets: list) -> list:
    '''
    Takes a list of dictionary objects representing tweets. Returns a binary feature of whether or not the tweet is a reply

    :param tweets: The list of tweets, where each tweet is a dictionary
    
    :return: list of binary elements; 1 = is a reply, 0 = not a reply
    '''
    
    return[1 if t['in_reply_to_user_id'] != None else 0 for t in tweets]

def length(tweets: list) -> list:
    '''
    Takes a list of dictionary objects representing tweets. Returns the number of tokens in the tweet

    :param tweets: The list of tweets, where each tweet is a dictionary

    :return: list of lengths of tweets
    '''

    return [len(t['text'].strip().split()) for t in tweets]

def retweet(tweets: list) -> list:
    '''
    Takes in a list of dictionary objects representing tweets.  Returns binary feature of whether or not the tweet is a retweet

    :param tweets: The list of tweets, where each tweet is a dictionary

    :return: list of binary elements; 1 = is a retweet, 0 = not a retweet
    '''
    return [1 if t['retweeted'] else 0 for t in tweets]

def num_retweets(tweets: list) -> list:
    '''
    Takes in a list of dictionary objects representing tweets.  Returns the number of times this tweet has been retweeted

    :param tweets: The list of tweets, where each tweet is a dictionary

    :return: list of counts of retweets of the tweets
    '''
    return [t['retweet_count'] for t in tweets]

def author(tweets: list) -> list:
    '''
    Takes in a list of dictionary objects representing tweets.  Returns the author id of the tweet

    :param tweets: The list of tweets, where each tweet is a dictionary

    :return: list of author ids from the tweets
    '''
    return [t['user']['id'] for t in tweets]

if __name__ == '__main__':
    tweets = [{'text': 'hello #whatup #you #me', 'in_reply_to_user_id': None, 'retweeted': True, 'retweet_count': 1000, 'user': {'id': 1234}}]
    
    #test num_hashtags
    assert(num_hashtags(tweets)[0] == 3)
    
    #test word_counts
    assert(word_counts(tweets)[0].nnz == 4)
    assert(len(word_counts(tweets)[1]) == 4)

    #test reply
    assert(reply(tweets)[0] == 0)

    #test length
    length_feat = length(tweets)
    assert(length_feat[0] == 4)
    
    #test tf_idf
    assert(tf_idf(tweets)[0].nnz == 4)
    assert(len(tf_idf(tweets)[1]) == 4)

    #test retweet
    assert(retweet(tweets)[0] == 1)

    #test num_retweets
    assert(num_retweets(tweets)[0] == 1000)

    #test author
    assert(author(tweets)[0] == 1234)
    
