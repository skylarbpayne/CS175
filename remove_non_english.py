from langdetect import detect
import nltk

names = [name.lower() for name in nltk.corpus.names.words()]
english_vocab = [w.lower() for w in nltk.corpus.words.words()]
chat_vocab = [w.lower() for w in nltk.corpus.nps_chat.words()]

full_vocab = set(english_vocab + names + chat_vocab)

def remove_non_english(tweets: list, thres: float = 0.5) -> list:
    '''
        tweets is a list of dictionaries representing tweet objects
        thres is a threshold that defines what percentage of words in a tweet must be english for it to be kept [0,1]
            a tweet is kept if pct_eng >= thres
        The output of this transformer is a list with all the non-english words removed.
    '''
    pct_eng = [sum([1 for t in tweet['text'].split() if t.strip('#').lower() in full_vocab])/len(tweet) for tweet in tweets]
    return [tweet for i,tweet in enumerate(tweets) if pct_eng[i] >= thres]
    
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

def remove_non_english2(tweets: list) -> list:
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
