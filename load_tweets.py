import json

def load_tweets(fn: str) -> list:
    with open(fn) as f:
        content = f.readlines()
        return [json.loads(s) for s in content]
