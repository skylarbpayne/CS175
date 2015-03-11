from langdetect import detect

def detect_wrapper(x: str) -> bool:
    '''
    Since langdetect.detect can throw exceptions, this will let us use detect in a list comprehension and handle the exceptions

    :param x: the string to detect the language of
    
    :return : True if english, False otherwise
    '''
    try:
        return detect(x) == 'en'
    except:
        return True

def remove_non_english(tweets: list) -> list:
    '''
        tweets is a list of dictionaries representing tweet objects
        thres is a threshold that defines what percentage of words in a tweet must be english for it to be kept [0,1]
            a tweet is kept if pct_eng >= thres
        The output of this transformer is a list with all the non-english words removed.
    '''
    return [tweet for tweet in tweets if detect_wrapper(tweet['text'])]
    

if __name__ == '__main__':
    '''This runs some simple unit tests to ensure that the transformer works correctly'''

    test_tweets = [{'text':'hey there jed'},{'text':'lwejk a;welkfj aw;elfj'}]
    t1 = remove_non_english(test_tweets, 1)
    t2 = remove_non_english(test_tweets, 0)
    print('Contents of test_tweets:', test_tweets)
    assert(len(t1) == 1)
    print('t1:', t1)
    assert(len(t2) == 2)
    print('t2:', t2)
