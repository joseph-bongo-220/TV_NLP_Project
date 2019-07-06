import NLP
from NLP import JBRank
import json
from app_config import get_config
from Scraper import Genius_TV_Scraper
import pickle
from Scraper import correct_characters
import time

config = get_config()

if __name__ == '__main__':
    shows = [x for x in config.keys() if x!= 'app']
    result_dict = {}
    for show in shows:
        show_dict={}
        pick = config["app"]["pickle"]
        episodes = NLP.process_episodes(show, Pickle=pick)
        print("episodes processed")
        start = time.time()
        ngrams = config["app"]["ngrams"]
        ep_rank=JBRank(docs=episodes, ngrams=ngrams)
        ep_rank.run()
        end = time.time()
        print(str(end-start))
        show_dict.update({"episode_keyphrases": ep_rank.final_rankings})

        for i in list(ep_rank.final_rankings.keys()):
            print(i)
            print(ep_rank.final_rankings[i])

        print ("*"*50)

        num_char = config[show]["num_characters"]
        characters=NLP.process_characters(show, num_char=num_char, Pickle=pick)
        print("chars processed")
        start = time.time()
        char_rank=JBRank(docs=characters, ngrams=ngrams)
        char_rank.run()
        end = time.time()
        print(str(end-start))
        show_dict.update({"character_keyphrases": char_rank.final_rankings})

        ep_algs = SemanticAlgos(episodes, show=show)
        show_dict.update({"episode_text_similarity": ep_algs.text_similarity()})
        show_dict.update({"episode_text_summarization": ep_algs.graph_text_summarization()})

        char_algs = SemanticAlgos(characters, show=show)
        show_dict.update({"character_dialogue_similarity": char_algs.text_similarity()})
        show_dict.update({"character_dialogue_summarization": char_algs.graph_text_summarization()})

        result_dict.update({show: show_dict})

        for i in list(char_rank.final_rankings.keys()):
            print(i)
            print(char_rank.final_rankings[i])