import NLP
from NLP import JBRank
import json
from app_config import get_config
from Scraper import Genius_TV_Scraper
import pickle
from Scraper import correct_characters

config = get_config()

if __name__ == '__main__':
    show="Game of thrones"
    pick = config["app"]["pickle"]
    episodes = NLP.process_episodes(show, Pickle=pick)

    ngrams = config["app"]["ngrams"]
    ep_rank=JBRank(docs=episodes, ngrams=ngrams)
    ep_rank.run()

    for i in list(ep_rank.final_rankings.keys()):
        print(i)
        print(ep_rank.final_rankings[i])

    print ("*"*50)

    num_char = config[show]["num_characters"]
    characters=NLP.process_characters(show, num_char=num_char, Pickle=pick)

    char_rank=JBRank(docs=characters, ngrams=ngrams)
    char_rank.run()

    for i in list(char_rank.final_rankings.keys()):
        print(i)
        print(char_rank.final_rankings[i])