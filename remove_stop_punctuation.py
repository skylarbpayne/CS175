import nltk
import string
punc = string.punctuation
english_stop_words = nltk.corpus.stopwords.words('english') + ['']

def remove_stop_punctuation(tokenized_tweets: list) -> list:
    '''
        tweets is a list of tokenized tweets (list of list of tokens).
        The output of this transformer is a list with all the stop words and punctuation removed.
    '''
    return[[token.strip(punc) for token in tweet if token.strip(punc).lower() not in english_stop_words] for tweet in tokenized_tweets]
    
    
if __name__ == '__main__':
    '''This runs some simple unit tests to ensure that the transformer works correctly'''

    test_tweets = [['Hey','There,','!'],['apple','orange','test'],['Website.']]
    t1 = remove_stop_punctuation(test_tweets)
    print('Contents of test_tweets:', test_tweets)
    assert(len(t1[0]) == 1)
    print('t1[0]:', t1[0])
    assert(len(t1[1]) == 3)
    print('t1[1]:', t1[1])
    assert(len(t1[2]) == 1)
    assert(t1[2][0] == 'Website')
    print('t1[2]:', t1[2])