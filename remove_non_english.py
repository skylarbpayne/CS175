import nltk
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

def remove_non_english(tokenized_tweets: list, thres: float) -> list:
    '''
        tweets is a list of tokenized tweets (list of list of tokens).
        thres is a threshold that defines what percentage of words in a tweet must be english for it to be kept [0,1]
            a word is kept if pct_eng >= thres
        The output of this transformer is a list with all the non-english words removed.
    '''
    pct_eng = [sum([1 for token in tweet if token.lower() in english_vocab])/len(tweet) for tweet in tokenized_tweets]
    return [tweet for i,tweet in enumerate(tokenized_tweets) if pct_eng[i] >= thres]
    

if __name__ == '__main__':
    '''This runs some simple unit tests to ensure that the transformer works correctly'''

    test_tweets = [['hey','there','jed'],['lwejk','a;welkfj','aw;elfj']]
    t1 = remove_non_english(test_tweets, 1)
    t2 = remove_non_english(test_tweets, 0)
    print('Contents of test_tweets:', test_tweets)
    assert(len(t1) == 1)
    print('t1:', t1)
    assert(len(t2) == 2)
    print('t2:', t2)
