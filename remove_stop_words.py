import nltk
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
    
if __name__ == '__main__':
    '''This runs some simple unit tests to ensure that the transformer works correctly'''

    test_tweets = [{'text':'Hey There'}, {'text':'apple orange test'}, {'text':'Website', 'in_reply_to_user_id': None}]
    t1 = remove_stop_punctuation(test_tweets)
    print('Contents of test_tweets:', test_tweets)
    assert(len(t1[0]['text'].split()) == 1)
    print('t1[0]:', t1[0])
    assert(len(t1[1]['text'].split()) == 3)
    print('t1[1]:', t1[1])
    assert(len(t1[2]['text'].split()) == 1)
    assert(len(t1[2]) == 2)
    print('t1[2]:', t1[2])  
