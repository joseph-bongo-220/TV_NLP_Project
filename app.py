from flask import Flask, jsonify, render_template
from app_config import get_config
from run import get_NLP_results

config = get_config()
results = get_NLP_results()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    shows = ["--SELECT A SHOW--"]+[x for x in config.keys() if x!= 'app']
    return(render_template('index.html', shows=shows))

@app.route('/<show>', methods=['GET'])
def get_show_info(show):
    shows = ["--SELECT A SHOW--"]+[x for x in config.keys() if x!= 'app']
    return(render_template('index_show.html', shows=shows, show_selected=show, seasons={"seasons":["All"]+[x+1 for x in range(config[show]["seasons"])]}))

@app.route('/<show>/<episode>', methods=['GET', 'POST'])
def get_episode_info(show, episode):
    episode_keyphrases = results[show]["episode_keyphrases"][episode]
    episode_text_similarity = results[show]["episode_text_similarity"][episode]
    episode_text_summarization = results[show]["episode_text_summarization"][episode]

@app.route('/<show>/<character>', methods=['GET', 'POST'])
def get_character_info(show, character):
    character_keyphrases = results[show]["character_keyphrases"][character]
    character_dialogue_similarity = results[show]["character_dialogue_similarity"][character]
    character_dialogue_summarization = results[show]["character_dialogue_summarization"][character]

if __name__ == '__main__':
    app.run(debug=config["app"]["debug"])