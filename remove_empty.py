def remove_empty(tweets: list) -> list:
    '''
    Removes any tweets with empty text

    :param tweets: a list of dictionary objects representing tweets

    :return: tweets with all empty tweets removed
    '''

    return [t for t in tweets if len(t['text'].strip()) > 0]
