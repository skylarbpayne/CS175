from load_auth import load_auth
import sys
import tweepy
import json

class ExtractionListener(tweepy.StreamListener):
    ''' Listens for tweets / errors'''

    tweets = []
    def on_data(self, data):
        ''' Gets data from the stream to build up a list of tweets from the stream'''
        self.tweets.append(data)
        return True
    def on_error(self, status):
        '''  Prints out errors should they occur'''
        print('Error:', status)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Not enough arguments. Please provide credential file and output file!")
        exit()
    
    auth_data = load_auth(sys.argv[1]) 
    auth = tweepy.OAuthHandler(auth_data['consumer_key'], auth_data['consumer_secret'])
    auth.set_access_token(auth_data['access_token'], auth_data['access_token_secret'])
    api = tweepy.API(auth)

    l = ExtractionListener()
    stream = tweepy.Stream(auth, l)
    print('Begin stream')
    try:
        stream.sample()
    except KeyboardInterrupt:
        pass
    print('\n%d tweets downloaded' % len(l.tweets))
    
    with open(sys.argv[2], 'w') as output:
        for tw in l.tweets:
            output.write(tw)
    print('Tweets saved to %s' % sys.argv[2])
    
    print('Getting trends')
    trends = api.trends_place(1) #1 is ID for worldwide
    out = sys.argv[2].split('.')
    out = '.'.join(out[0:-1] + ['trends'] + [out[-1]])
    with open(out, 'w') as trends_file:
        trends_file.write(json.dumps(trends))
    print('Finishing writing worldwide trends to %s' % out)
