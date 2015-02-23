import string
punctuation = string.punctuation.replace('#', '')

def remove_stop_punctuation(tweets: list) -> list:
    '''
        tweets is a list of dictionaries representing tweet objects
        The output of this transformer is a list with all the punctuation removed.
    '''
    for tweet in tweets:
        tweet['text'] = ' '.join([token.strip(punctuation) for token in tweet['text'].split() if token.lower() not in punctuation])
    return tweets
    
if __name__ == '__main__':
    '''This runs some simple unit tests to ensure that the transformer works correctly'''

    test_tweets = [{'text':'Hey There !'}, {'text':'#apple orange test'}, {'text':'Website.', 'in_reply_to_user_id': None}]
    t1 = remove_stop_punctuation(test_tweets)
    print('Contents of test_tweets:', test_tweets)
    assert(len(t1[0]['text'].split()) == 2)
    print('t1[0]:', t1[0])
    assert(len(t1[1]['text'].split()) == 3)
    assert(t1[1]['text'][0] == '#')
    print('t1[1]:', t1[1])
    assert(len(t1[2]['text'].split()) == 1)
    assert(t1[2]['text'] == 'Website')
    assert(len(t1[2]) == 2)
    print('t1[2]:', t1[2])