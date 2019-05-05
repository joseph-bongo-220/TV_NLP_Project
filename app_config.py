import json

def get_config():
    with open('config.json') as f:
        config = json.load(f)
    return config