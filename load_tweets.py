import json

def load_tweets(fn: str) -> list:
    with open(fn) as f:
        content = f.readlines()
        return [t for t in [json.loads(s) for s in content] if 'text' in t]
