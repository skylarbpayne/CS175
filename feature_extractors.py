from sklearn.feature_extraction.text import CountVectorizer

def word_counts(tweets: list) -> (np.ndarray, list):
    '''
    Takes a list of dictionary objects representing tweets. Returns a sparse word count matrix of size (num tweets) by (num words)

    :param tweets: The list of tweets, where each tweet is a dictionary
    :return: tuple of (count matrix, vocabulary list)
    '''
    
    cv = CountVectorizer()
    text = [t['text'] for t in tweets]
    return cv.fit_transform(text), cv.get_feature_names()

def num_hashtags(tweets: list) -> list:
    '''
    Takes a list of dictionary objects representing tweets. Returns the number of hashtag counts for each tweet

    :param tweets: The list of tweets, where each tweet is a dictionary
    :return: list of hashtag counts
    '''

    return [t['text'].count('#') for t in tweets]

if __name__ == '__main__':
    tweets = [{'text': 'hello #whatup #you #me'}]
    
    assert(num_hashtags(tweets)[0] == 3)

    assert(word_counts(tweets)[0].nnz == 4)
    assert(len(word_counts(tweets)[1]) == 4)
