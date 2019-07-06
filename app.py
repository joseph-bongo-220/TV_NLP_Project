from flask import Flask, jsonify, render_template
from app_config import get_config

config = get_config()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    shows = ["--SELECT A SHOW--"]+[x for x in config.keys() if x!= 'app']
    return(render_template('index.html', shows=shows))

@app.route('/<show>', methods=['GET'])
def get_show_info(show):
    shows = ["--SELECT A SHOW--"]+[x for x in config.keys() if x!= 'app']
    return(render_template('index_show.html', shows=shows, show_selected=show, seasons={"seasons":[x+1 for x in range(config[show]["seasons"])]+["All"]}))

@app.route('/<show>/<episode>', methods=['GET', 'POST'])
def get_episode_info(show, episode):
    pass

@app.route('/<show>/<character>', methods=['GET'])
def get_character_info(show, character):
    pass

if __name__ == '__main__':
    app.run(debug=config["app"]["debug"])