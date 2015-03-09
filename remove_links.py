import string
import re

link_re = '^[a-zA-Z0-9\-\.]+\.(com|org|net|mil|edu|COM|ORG|NET|MIL|EDU)$'

def remove_links(tweets: list) -> list:
    '''
    tweets is a list of dictionaires representing tweet objects
    The output of this transformer is a list with all links removed.
    '''
    for tweet in tweets:
        tweet["text"] = ' '.join([token for token in tweet['text'].split() if not re.match(link_re, token)])
    return tweets
                             
if __name__ == '__main__':
    '''This runs some simple unit tests to ensure that the transformer works correctly'''

    test_tweets = [{'text':'Hey There !'}, {'text':'#apple orange test'}, {'text':'www.facebook.com need a visual update!', 'in_reply_to_user_id': None}, {'text': 'I think apple.net is cool'}]
    t1 = remove_links(test_tweets)

    print('Contents of test_tweets: ', test_tweets)
    assert(t1[2]['text'] == 'need a visual update!')
    assert(len(t1[2]['text'].split()) == 4)
    assert(t1[3]['text'] == 'I think is cool')
    assert(len(t1[3]['text'].split()) == 4)
