import NLP
from NLP import JBRank
import json
from app_config import get_config
from Scraper import Genius_TV_Scraper

config = get_config()

if __name__ == '__main__':
    pick = config["app"]["pickle"]
    episodes = NLP.process_episodes("Game of thrones", Pickle=pick)

    ngrams = config["app"]["ngrams"]
    X=JBRank(docs=episodes, ngrams=ngrams)
    X.stats()

    measure = config["app"]["measure"]
    X.graph(measure=measure)

    for i in list(X.final_rankings.keys()):
        print(X.final_rankings[i])
