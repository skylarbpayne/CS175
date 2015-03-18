import nltk
import string
from langdetect import detect
import re

def remove_digits(tweets: list) -> list:
    '''
    Removes any text consisting of digits.

    :param tweets: a list of dictionary objects representing tweets

    :return: a list of tweets with digits removed
    '''

    text = [tw['text'].split() for tw in tweets]
    text = [' '.join([tok for tok in t if tok.isalpha()]) for t in text]
    for tw,txt in zip(tweets,text):
        tw['text'] = txt 

    return tweets


def remove_empty(tweets: list) -> list:
    '''
    Removes any tweets with empty text

    :param tweets: a list of dictionary objects representing tweets

    :return: tweets with all empty tweets removed
    '''

    return [t for t in tweets if len(t['text'].strip()) > 0]

link_re = '^[a-zA-Z0-9\-\.]+\.(com|org|net|mil|edu|COM|ORG|NET|MIL|EDU)$'

def remove_links(tweets: list) -> list:
    '''
    tweets is a list of dictionaires representing tweet objects
    The output of this transformer is a list with all links removed.
    '''
    for tweet in tweets:
        tweet["text"] = ' '.join([token for token in tweet['text'].split() if not re.match(link_re, token)])
    return tweets

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

punctuation = string.punctuation #.replace('#', '')

def remove_punctuation(tweets: list) -> list:
    '''
        tweets is a list of dictionaries representing tweet objects
        The output of this transformer is a list with all the punctuation removed.
    '''
    for tweet in tweets:
        tweet['text'] = ' '.join([token.strip(punctuation) for token in tweet['text'].split() if token.lower() not in punctuation])
    return tweets

def remove_small_tokens(tweets: list, k: int = 2) -> list:
    '''
    Removes any token that has length less than or equal to k.

    :param tweets: a list of dictionary objects representing tweets

    :return : a list of tweets with some tokens removed.
    '''

    for tweet in tweets:
        tweet['text'] = ' '.join([token for token in tweet['text'].strip().split() if len(token) > k]) 
    return tweets

twitter_terms = [ 'rt' , 'follow' ]
daddy_yankee = ['si', 'que', 'sígueme', 'te', 'di', 'da']

english_stop_words = nltk.corpus.stopwords.words('english') + twitter_terms + daddy_yankee 

def remove_stop_words(tweets: list) -> list:
    '''
        tweets is a list of dictionaries representing tweet objects
        The output of this transformer is a list with all the stop words removed.
    '''
    for tweet in tweets:
        tweet['text'] = ' '.join([token for token in tweet['text'].split() if token.lower() not in english_stop_words])
    return tweets