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
