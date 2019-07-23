import NLP
from NLP import JBRank, SemanticAlgos
import json
from app_config import get_config
from Scraper import Genius_TV_Scraper, correct_characters
import pickle
from Scraper import correct_characters
import time

config = get_config()

def get_NLP_results():
    shows = [x for x in config.keys() if x!= 'app']
    result_dict = {}
    for show in shows:
        show_dict={}
        pick = config["app"]["pickle"]
        episodes, season_dict = NLP.process_episodes(show, Pickle=pick)
        show_dict.update({"seasons": season_dict})
        ngrams = config["app"]["ngrams"]
        ep_rank=JBRank(docs=episodes, ngrams=ngrams)
        ep_rank.run()
        show_dict.update({"episode_keyphrases": ep_rank.final_rankings})

        num_char = config[show]["num_characters"]
        characters=NLP.process_characters(show, num_char=num_char, Pickle=pick)
        print("chars processed")
        char_rank=JBRank(docs=characters, ngrams=ngrams)
        char_rank.run()
        show_dict.update({"character_keyphrases": char_rank.final_rankings})

        ep_algs = SemanticAlgos(episodes, show=show)
        start = time.time()
        show_dict.update({"episode_text_similarity": ep_algs.text_similarity()})
        end = time.time()
        print("Text similarity time " + str(end-start))
        start = time.time()
        show_dict.update({"episode_text_summarization": ep_algs.graph_text_summarization(doc_type="episodes")})
        end = time.time()
        print("Text summarization time " + str(end-start))

        char_algs = SemanticAlgos(characters, show=show)
        show_dict.update({"character_dialogue_similarity": char_algs.text_similarity()})
        show_dict.update({"character_dialogue_summarization": char_algs.graph_text_summarization(doc_type="chars")})

        result_dict.update({show: show_dict})

    return(result_dict)

if __name__ == '__main__':
    start = time.time()
    res = get_NLP_results()
    end = time.time()
    print(res)
    print("Total time " + str(end-start))
