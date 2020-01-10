import NLP
from NLP import JBRank, SemanticAlgos
import aws_functions as aws
import json
import pandas as pd
from app_config import get_config
from Scraper import Genius_TV_Scraper, correct_characters
import pickle
from Scraper import correct_characters
import time
import json
import os
import re

config = get_config()

def get_NLP_results(shows = [x for x in config.keys() if x not in ['app', "aws"]]):
    connection = aws.connect_to_rds(username = os.environ["RDS_USERNAME"], password = os.environ["RDS_PASSWORD"])
    result_dict = {}
    for show in shows:
        print(show)
        if show == "Game of thrones":
            decay=True
        else:
            decay=False
        show_dict={}
        pick = config["app"]["use_s3"]
        episodes, season_dict = NLP.process_episodes(show, DB=pick)
        print("episodes processed")
        show_dict.update({"seasons": season_dict})
        ngrams = config["app"]["JBRank"]["ngrams"]
        start = time.time()
        ep_rank=JBRank(docs=episodes, include_title=True, term_len_decay=decay, ngrams=ngrams)
        ep_rank.run()
        end = time.time()
        print("JBRank time " + str(end-start))
        show_dict.update({"episode_keyphrases": ep_rank.final_rankings})

        num_char = config[show]["num_characters"]
        characters=NLP.process_characters(show, num_char=num_char, DB=pick)
        print("chars processed")
        start = time.time()
        char_rank=JBRank(docs=characters, include_title=False, term_len_decay=decay, ngrams=ngrams)
        char_rank.run()
        end = time.time()
        print("JBRank time " + str(end-start))
        show_dict.update({"character_keyphrases": char_rank.final_rankings})

        ep_algs = SemanticAlgos(episodes, doc_type="episodes", sent_threshold=config["app"]["text_summarization"]["sentence_similarity_threshold"],
         show=show, clean_sents = config["app"]["clean_sents"], clean_docs = config["app"]["clean_docs"])
        start = time.time()
        show_dict.update({"episode_text_similarity": ep_algs.text_similarity()})
        end = time.time()
        print("Text similarity time " + str(end-start))
        start = time.time()
        summs = ep_algs.graph_text_summarization()

        for key, val in summs.items():
            query = """SELECT character_name, narration FROM {show}
            WHERE episode = {episode}""".format(show = re.sub(" ", "_", show).lower(), episode = "'"+key+"'")
            df = pd.read_sql(query, connection)
            new_val = NLP.add_ep_speakers(df, val)
            summs[key] = new_val
        show_dict.update({"episode_text_summarization": summs})
        end = time.time()
        print("Text summarization time " + str(end-start))

        char_algs = SemanticAlgos(characters, doc_type="chars", show=show)
        show_dict.update({"character_dialogue_similarity": char_algs.text_similarity()})

        result_dict.update({show: show_dict})

    return(result_dict)

if __name__ == '__main__':
    start = time.time()
    res = get_NLP_results(shows = ["The office us"])
    end = time.time()
    print("Total time " + str(end-start))
