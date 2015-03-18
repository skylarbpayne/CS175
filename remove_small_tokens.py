'''
Often, there are very small words that are usually never trends. remove_small_tokens helps that.
'''

def remove_small_tokens(tweets: list, k: int = 2) -> list:
    '''
    Removes any token that has length less than or equal to k.

    :param tweets: a list of dictionary objects representing tweets

    :return : a list of tweets with some tokens removed.
    '''

    for tweet in tweets:
        tweet['text'] = ' '.join([token for token in tweet['text'].strip().split() if len(token) > k]) 
    return tweets

if __name__ == '__main__':
    tweets = [{'text': 'hi my name is Jerry'}]
    assert(remove_small_tokens(tweets)[0]['text'] == 'name Jerry')
