import nltk
import string
punc = string.punctuation.replace('#', '')
english_stop_words = nltk.corpus.stopwords.words('english') + ['']

def remove_stop_punctuation(tweets: list) -> list:
    '''
        tweets is a list of dictionaries representing tweet objects
        The output of this transformer is a list with all the stop words and punctuation removed.
    '''
    for tweet in tweets:
        tweet['text'] = ' '.join([token.strip(punc) for token in tweet['text'].split() if token.strip(punc).lower() not in english_stop_words])
    return tweets
    
if __name__ == '__main__':
    '''This runs some simple unit tests to ensure that the transformer works correctly'''

    test_tweets = [{'text':'Hey There, !'}, {'text':'#apple orange test'}, {'text':'Website.', 'in_reply_to_user_id': None}]
    t1 = remove_stop_punctuation(test_tweets)
    print('Contents of test_tweets:', test_tweets)
    assert(len(t1[0]['text'].split()) == 1)
    print('t1[0]:', t1[0])
    assert(len(t1[1]['text'].split()) == 3)
    print('t1[1]:', t1[1])
    assert(len(t1[2]['text'].split()) == 1)
    assert(t1[2]['text'] == 'Website')
    assert(len(t1[2]) == 2)
    print('t1[2]:', t1[2])