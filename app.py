from flask import Flask, jsonify, render_template
from app_config import get_config
from run import get_NLP_results
import time

app = Flask(__name__)

config = get_config()
start = time.time()
results = get_NLP_results()
end = time.time()
print("Total time " + str(end-start))

shows = ["--SELECT A SHOW--"]+[x for x in config.keys() if x not in ['app', 'aws']]

@app.route('/', methods=['GET'])
def index():
    return(render_template('index.html', shows=shows))

@app.route('/show=<show>', methods=['GET'])
def get_show_info(show):
    characters = [x for x in results[show]["character_keyphrases"].keys()]
    return(render_template('index_show.html', shows=shows, show_selected=show, characters=characters, seasons={"seasons":[str(x+1) for x in range(config[show]["seasons"])]+["All"]}))

@app.route('/show=<show>/season=<season>', methods=['GET', 'POST'])
def get_episodes(show="Game of thrones", season="All"):
    episodes = results[show]["seasons"][str(season)]

    return(render_template('index_show_episodes.html', shows=shows, show_selected=show, season_selected=str(season), 
    episodes=episodes, seasons={"seasons":[str(x+1) for x in range(config[show]["seasons"])]+["All"]}))

@app.route('/show=<show>/season=<season>/episode=<episode>', methods=['GET', 'POST'])
def get_episode_info(show, season, episode):
    episodes = results[show]["seasons"][str(season)]
    episode_keyphrases = results[show]["episode_keyphrases"][episode]
    episode_text_similarity = results[show]["episode_text_similarity"][episode]
    episode_text_summarization = results[show]["episode_text_summarization"][episode]

    return(render_template('index_show_info_eps.html', shows=shows, show_selected=show, entity=episodes, entity_selected=episode, 
    keyphrases = episode_keyphrases, text_similarity = episode_text_similarity, text_summarization = episode_text_summarization,
    seasons={"seasons":[str(x+1) for x in range(config[show]["seasons"])]+["All"]}))

@app.route('/show=<show>/char=<character>', methods=['GET', 'POST'])
def get_character_info(show, character):
    characters = [x for x in results[show]["character_keyphrases"].keys()]
    character_keyphrases = results[show]["character_keyphrases"][character]
    character_dialogue_similarity = results[show]["character_dialogue_similarity"][character]

    return(render_template('index_show_info.html', shows=shows, show_selected=show, entity=characters, entity_selected=character, 
    keyphrases = character_keyphrases, text_similarity = character_dialogue_similarity,
    seasons={"seasons":[str(x+1) for x in range(config[show]["seasons"])]+["All"]}))

if __name__ == '__main__':
    app.run(debug=config["app"]["debug"], use_reloader=config["app"]["use_reloader"])