# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 01:29:33 2018

@author: Joe
"""

"""Use My Own Algorithm to Extract Keyphrases"""
from nltk.tokenize import word_tokenize
import networkx as nx
import pickle
import pandas as pd 
import inflect
from collections import Counter
import re
import numpy as np
from Scraper import Genius_TV_Scraper
from Scraper import correct_characters
import math
import sys
from helper import second, flatten, sort_dict, smallest_distance
from text_cleaning import clean_text, remove_stopwords, replace_numbers, get_contractions
from app_config import get_config
import os

config = get_config()

def pickle_data_frames(show):
    """Scrapes Genius scripts and saves them as Pickle files to be accessed later and to be easily called by API."""
    pickle_files = [f for f in os.listdir('.') if os.path.isfile(f) and re.search(".pkl",f) is not None]
    path = config[show]["pickle_path"]

    if path in pickle_files:
        pass
    else:
        Scraper = Genius_TV_Scraper(show=show)
        data = Scraper.get_scripts()
        data.to_pickle(path)

def process_episodes(show, seasons=None, Pickle=False):
    if Pickle:
        print("Pulling Pickle Data")
        pickle_path = config[show]["pickle_path"]
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

    else:
        print("Gathering Data from Genius.com")
        # initialize scraper for desire show and seasons (default for seasons is all available on Genius)
        scraper = Genius_TV_Scraper(show=show, seasons=seasons)

        #scrape previously specified show and seasons
        data = scraper.get_scripts()

    #get episodes from data
    data=correct_characters(data, show)
    
    z = []
    for i in list(data["Episode"]):
        if i not in z:
            z.append(i)
    
    docs=[]
    for episode in z:
        df = data[data.Episode == episode]
        doc = " ".join(list(df["Line"]))
        doc_dict = {episode: doc}
        docs.append(doc_dict)

    return docs

def process_characters(show, seasons=None, num_char=50, Pickle=False):
    if Pickle:
        print("Pulling Pickle Data")
        pickle_path = config[show]["pickle_path"]
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

    else:
        print("Gathering Data from Genius.com")
        # initialize scraper for desire show and seasons (default for seasons is all available on Genius)
        scraper = Genius_TV_Scraper(show=show, seasons=seasons)

        #scrape previously specified show and seasons
        data = scraper.get_scripts()
    
    #get characters from data
    data=correct_characters(data, show)

    remove_char=config[show]["remove_chars"]
    char_list = [x for x in data["Character_Name"] if x not in remove_char]
    char_counter=Counter(char_list)
    z = [x[0] for x in sorted(char_counter.items(), key=lambda x: x[1], reverse=True)][0:num_char]

    docs=[]
    for char in z:
        df = data[data.Character_Name == char]
        char_text = " ".join(list(df["Line"]))
        doc_dict = {char: char_text}
        docs.append(doc_dict)

    return docs

class JBRank(object):
    def __init__(self, docs, ngrams=[1,2,3,4,5,6], position_cutoff=5000, graph_cutoff=500, take_top=50):
        cList = get_contractions()
        self.TF_list=[]
        self.tokenized_docs=[]
        self.doc_titles=[]
    
        for doc_dict in docs:
            self.doc_titles.append(list(doc_dict.keys())[0])
            doc = list(doc_dict.values())[0]
            doc = doc.lower()
            
            for x in list(cList.keys()):
                doc = re.sub(x, cList[x], doc)
            
            # Replace other uncommon contractions (mainly from GOT)
            doc = re.sub("g'", "good ", doc)
            doc = re.sub("m'", "my ", doc)
            doc = re.sub("d'", "do ", doc)
            
            # Tokenize document
            word_list = word_tokenize(doc)
            
            # Clean Text
            word_list = clean_text(word_list)
            
            # Replace numbers with numerical representation
            word_list = replace_numbers(word_list)
            
            self.tokenized_docs.append(word_list)
            
            BOW = self.get_BOW_TF(word_list)
            
            self.grams=[x for x in ngrams if x >=2 and x<=6]
            
            for size in self.grams:
                tokenized_ngrams = ["_".join(word_list[x:x+size]) for x in range(len(word_list)) if len(word_list[x:x+size])== size]
                NBOW = self.get_BOW_TF(tokenized_ngrams)
                BOW.update(NBOW)
            
            self.TF_list.append(BOW)
            
            #self.TF_list=self.subsum_correction(TF_list)
            
        IDF_list = self.get_BOW_IDF(self.TF_list)
        
        self.tf_idf_list = self.get_TFIDF_info(tf_dicts=self.TF_list, idf_dict=IDF_list)
        
        self.position_cutoff=position_cutoff
        self.graph_cutoff=graph_cutoff
        self.take_top=take_top
        
    def stats(self):
        PFO_factor={}
        tl_factor={}
        for doc in self.tokenized_docs:
            PFO_factor.update(self.get_PFO(doc, self.position_cutoff))
            tl_factor.update(self.get_TL(doc))
            for size in self.grams:
                tokenized_ngrams = ["_".join(doc[x:x+size]) for x in range(len(doc)) if len(doc[x:x+size])== size]
                PFO_factor.update(self.get_PFO(tokenized_ngrams, self.position_cutoff))
                tl_factor.update(self.get_TL(tokenized_ngrams))
                
        self.stat_ranking= self.tf_idf_list.copy()
        
        for i in range(len(self.stat_ranking)):
            for key, val in self.stat_ranking[i].items():
                self.stat_ranking[i][key]=val*PFO_factor[key]*tl_factor[key]
            self.stat_ranking[i] = sort_dict(self.stat_ranking[i])[:self.take_top]
            self.stat_ranking[i] = {key:value for key,value in self.stat_ranking[i]}
            
    def graph(self, measure="pagerank"):
        self.final_rankings = {}
        for i in range(len(self.stat_ranking)):        
            ranking = self.stat_ranking[i]
            doc = self.tokenized_docs[i]
            words = list(ranking.keys())
            G=nx.Graph()
            G.add_nodes_from(words)
            
            #get edges of graphs
            indices_dict={}
            for word in words:
                indices_dict.update({word:self.get_inds_for_gram(word, doc)})

            weights_df=self.get_edge_weights(indices_dict, self.graph_cutoff)
            for word1 in words:
                for word2 in words:
                    if weights_df[word1][word2]==0 or math.isnan(weights_df[word1][word2]):
                        pass
                    else:
                        G.add_edge(word1, word2, weight=weights_df[word1][word2]*ranking[word1]*ranking[word2])
            if measure == "pagerank":
                gr_dict=nx.pagerank(G)
            elif measure == "betweenness centrality":
                gr_dict=nx.betweenness_centrality(G, weight="weight")
            elif measure == "load centrality":
                gr_dict=nx.load_centrality(G, weight="weight")
            gr_dict_sorted=sort_dict(gr_dict)
            self.final_rankings.update({self.doc_titles[i]: {key:value for key,value in gr_dict_sorted}})

    def get_inds_for_gram(self, word, tokenized_doc):
        split_=word.split("_")
        grams=[]
        for wordy in split_:
            grams.append([i for i,e in enumerate(tokenized_doc) if re.search(wordy, tokenized_doc[i]) is not None])

        inds=[]
        for j in grams[0]:
            temp = [j]
            for i in range(len(grams)-1):
                if j+i+1 in grams[i+1]:
                    temp.append(j+i+1)
            if len(temp)==len(grams):
                inds.append(temp)
        return inds
    
    def get_edge_weights(self, wi_dict, cutoff):
        edge_df=pd.DataFrame(index=list(wi_dict.keys()), columns=list(wi_dict.keys()))
        for word, indices in wi_dict.items():
            for word2, indices2 in wi_dict.items():
                dists = []
                if math.isnan(edge_df[word2][word])==False or word==word2:
                    pass 
                else:
                    len_word=len(re.findall("_",word))
                    len_word2=len(re.findall("_",word2))
                    ngram_factor = max([len([x for x in word.split("_") if x not in word2.split("_")]), len([x for x in word2.split("_") if x not in word.split("_")])])
                    for x in wi_dict[word]:
                        dists.extend([smallest_distance(x,y)+ngram_factor for y in wi_dict[word2] if smallest_distance(x,y) < cutoff and smallest_distance(x,y)!=0])
                    num=0
                    if len(dists)==0:
                        edge_df[word][word2]=0
                        edge_df[word2][word]=0
                    else:
                        for dist in dists:
                            num=num+np.log(cutoff/dist)
                        edge_df[word][word2]=num/len(dists)
                        edge_df[word2][word]=0
        return edge_df

    def get_BOW_TF(self, words):
        """returns a dictionary with the term frequency of all terms in a given document"""
        bagofwords = dict(Counter(words))
        return bagofwords

    def get_BOW_IDF(self, list_of_bows):
        """returns a dictionary with the inverse document frequency of all terms in a given corpus"""
        temp = []
        for i in list_of_bows:
            temp.append(i.keys())
        
        term_list = flatten(temp)
        
        idf_list=dict(Counter(term_list))
        
        for key, val in idf_list.items():
            if len(re.findall("_", key))<2:
                idf_list[key] = np.log(len(list_of_bows)/val)
                
            else:
                idf_list[key] = np.log(len(list_of_bows))
        
        return idf_list

    def get_TFIDF_info(self, tf_dicts, idf_dict):
        """takes in the term frequency dictionaries and IDF dictionary anc calculates TFIDF for every term in the corpus"""
        tf_idf_dict = tf_dicts.copy()
        for i in range(0,len(tf_idf_dict)):
            for key, val in tf_idf_dict[i].items():
                tf_idf_dict[i][key] = tf_idf_dict[i][key]*idf_dict[key]
        return tf_idf_dict
    
    def get_PFO(self, tokenized_doc, cutoff_position):
        """Get the position of first occurance for a given word in a document"""
        PFO_dict={}
        for i in range(1,len(tokenized_doc)+1):
            nums = [i for i in range(1,len(tokenized_doc)+1)]
            word = tokenized_doc[-i]
            if word in PFO_dict==True:
                PFO_dict[word]=nums[-i]
            else:
                PFO_dict.update({word:nums[-i]})
        
        for key, val in PFO_dict.items():
            PFO_dict[key] = np.log(cutoff_position/val)  
        return PFO_dict

    def get_TL(self, tokenized_doc):
        tl_dict={}
        for word in tokenized_doc:
            tl_dict.update({word:np.log(len(re.findall("_", word))+2)})
            
        return tl_dict

    def subsum_correction(self, tf_dict_list):
        for tf_dict in tf_dict_list:
            for key, val in tf_dict.items():
                if len(key.split(sep="_")) >= 2:
                    for word in key.split(sep="_"):
                        tf_dict[word]=tf_dict[word]-tf_dict[key]
                    
        return tf_dict_list

    def run(self):
        self.stats()
        measure = config["app"]["measure"]
        self.graph(measure=measure)

if __name__ == "__main__":
    with open('examples.pkl', 'rb') as f:
        docs = pickle.load(f)
    x=JBRank(docs=docs, ngrams=[1,2])
    x.run()