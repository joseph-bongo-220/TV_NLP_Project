# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 01:29:33 2018

@author: Joe
"""

"""Use My Own Algorithm to Extract Keyphrases"""
from nltk.tokenize import sent_tokenize, word_tokenize
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
from text_cleaning import *
from app_config import get_config
import os
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import boto3
from botocore.exceptions import ClientError, ParamValidationError
import aws_functions as aws
from copy import deepcopy
import psycopg2

import tensorflow as tf 
import tensorflow_hub as hub

# import config file
config = get_config()

def get_pickle_data_frames(show, seasons=None):
    """Scrapes Genius scripts and saves them as Pickle files to be accessed later and to be easily called by API."""
    # connect to postgres db
    connection = aws.connect_to_rds(username = os.environ["RDS_USERNAME"], password = os.environ["RDS_PASSWORD"])

    # load the script data from AWS
    data = aws.get_script_data(show=show, connection=connection, seasons=seasons)
    return data

def process_episodes(show, seasons=None, DB=True):
    """Takes dataframes containing script information and creates episode documents"""
    # S3 = True: Data will be pulled from pkl file in S3 bucket (will be changed to Postgres)
    if DB:
        data = get_pickle_data_frames(show=show, seasons=seasons)

    # S3 = False: Data will be scraped from Genius.com using the Genius_TV_Scraper object 
    # defined in Scraper.py
    else:
        print("Gathering Data from Genius.com")
        # initialize scraper for desire show and seasons (default for seasons is all available on Genius)
        scraper = Genius_TV_Scraper(show=show, seasons=seasons)

        #scrape previously specified show and seasons
        data = scraper.get_scripts()

        # add data to database
        aws.put_show_rds(data, show)

    # correct the character names using show character config and 
    # fuzzy matching and partial fuzzy matching using Levenshtein
    # distance
    data=correct_characters(data, show)
    
    # get unique episodes (LOL this is inefficient)
    z = []
    for i in list(data["episode"]):
        if i not in z:
            z.append(i)

    # initialize document dictionary
    docs={}

    # iterate over each episode available
    for episode in z:

        # subset data to only include
        df = data[data.episode == episode]

        # # if no line, include narration
        # for i in range(len(df)):
        #     if df["Line"].iloc[i]=='':
        #         print(df["Narration"].iloc[i])
        #         if "HBO" not in df["Narration"].iloc[i]:
        #             df["Line"].iloc[i]=df["Narration"].iloc[i]

        # add punctuation to lines when necessary
        lines = [punctuate_line(x) for x in list(df["line"])]

        # append the lines and update the dict: 
        # key = episode name
        # value = document
        doc = " ".join(lines)
        docs.update({episode: doc})

    # create dictionary to store all of the epsodes from a given season
    # "All" includes all available episodes
    season_dict = {"All": z}

    # seasons defaults to all seasons
    if seasons is None:
        seasons = [i+1 for i in range(config[show]["seasons"])]

    # get each season's episodes
    for i in seasons:
        x = []
        for j in list(data["episode"][data["season"]==i]):
            if j not in x:
                x.append(j)
        # add season to dict
        season_dict.update({str(i): x})

    return docs, season_dict

def process_characters(show, seasons=None, num_char=50, DB=True):
    """Takes dataframes containing script information and creates character documents"""
    # S3 = True: Data will be pulled from pkl file in S3 bucket (will be changed to Postgres)
    if DB:
        data = get_pickle_data_frames(show=show, seasons=seasons)

    # S3 = False: Data will be scraped from Genius.com using the Genius_TV_Scraper object 
    # defined in Scraper.py
    else:
        print("Gathering Data from Genius.com")
        # initialize scraper for desire show and seasons (default for seasons is all available on Genius)
        scraper = Genius_TV_Scraper(show=show, seasons=seasons)

        # scrape previously specified show and seasons
        data = scraper.get_scripts()
        aws.put_show_rds(data, show)
    
    # correct the character names using show character config and 
    # fuzzy matching and partial fuzzy matching using Levenshtein
    # distance
    data=correct_characters(data, show)

    # remove characters that are not relevant "MAN", etc.
    remove_char=config[show]["remove_chars"]

    # get top num_char characters by number of lines spoken
    char_list = [x for x in data["character_name"] if x not in remove_char]
    char_counter = Counter(char_list)
    z = [x[0] for x in sorted(char_counter.items(), key=lambda x: x[1], reverse=True)][0:num_char]

    # initialize documents dicitionary
    docs={}
    for char in z:
        # only use data from a given characters
        df = data[data.character_name == char]

        # get characters lines
        lines = [punctuate_line(x) for x in list(df["line"])]

        # create documentof character dialogue and update dictionary
        # key: Character Name
        # value: All Character Dialogue
        char_text = " ".join(lines)
        docs.update({char: char_text})
    return docs

def add_ep_speakers(ep_df, ep_summs):
    """adds sentence speaker to text summarizations"""
    # create column N_Lower to assure text matches sentences
    cList = get_contractions()
    cList.update({" g'": " good ",
        " m'": " my ",
        " d'": " do ", 
        " s ": "s ", 
        "[ ]{2,}": " "})
    for z in list(cList.keys()):
        ep_df["n_lower"] = [re.sub(z, cList[z], y) for y in ep_df["narration"]]
    
    # initialize result list
    results = []

    # list regex special charaters that need replacement
    regex_special_chars = ["\)", "\(", "\*", "\+", "\[", "\]", "\^", "\$"]

    # copy list of summary sentences
    ep_summs2 = ep_summs[:]

    # turn copy of list into valid regular expressions
    for i in regex_special_chars:
        ep_summs2 = [re.sub(i, i, x) for x in ep_summs2]

    # iterate over summary sentence indices
    for x in range(len(ep_summs)):
        # get the character name for the corresponding sentences
        temp = list(ep_df["character_name"][ep_df["n_lower"].str.contains(ep_summs2[x])].values)
        
        # TEMPORARY WORK AROUND
        # Add speaker(s) to original summary sentences
        if len(set(temp)) > 2:
            temp = ", ".join(temp)+": "+ep_summs[x]
        elif len(set(temp)) == 2:
            temp = " and ".join(temp)+": "+ep_summs[x]
        elif len(set(temp)) == 1:
            temp = temp[0]+": "+ep_summs[x]
        else:
            temp = ep_summs[x]
        results.append(temp) 
    return results

class JBRank(object):
    """Keyphrase Extraction Algorithm that I wrote based on unsupervised SGRank paper. More info will follow in README"""
    def __init__(self, docs, include_title=False, term_len_decay=True, ngrams=[1,2,3,4,5,6], position_cutoff=5000, graph_cutoff=500, take_top=50, show="Game of thrones", correct_subsum=False):
        """Parses provided documents and gathers the TF-IDF for each word for each document."""
        # confirm documents are in the proper format
        # Raise Error otherwise
        if isinstance(docs, dict) == False:
            raise TypeError("Documents must be in the form of a dictionary")
        
        # get contractions to be replaced
        cList = get_contractions()

        # initialize several attributes of objects
        self.TF_list=[]
        self.tokenized_docs=[]
        self.doc_titles=[]
        self.include_title = include_title
        self.term_len_decay = term_len_decay
    
        # iterate over the documents
        for key, value in docs.items():
            # save title
            title = key
            self.doc_titles.append(title)

            # coerce document to all lowercase
            doc = value
            doc = doc.lower()

            # add title to document
            if self.include_title:
                doc = re.sub("^season [^ ]* episode [^ ]* ", "", punctuate_line(title.lower())) + doc
            
            # Replace contractions
            for x in list(cList.keys()):
                doc = re.sub(x, cList[x], doc)
            
            # Replace other uncommon contractions (mainly from GOT)
            doc = re.sub(" g'", " good ", doc)
            doc = re.sub(" m'", " my ", doc)
            doc = re.sub(" d'", " do ", doc)
            doc = re.sub(" s ", "s ", doc)
            doc = re.sub("[ ]{2,}", " ", doc)
            
            # Tokenize document
            word_list = word_tokenize(doc)

            # Remove non-ASCII characters from list of tokenized words
            word_list = clean_text(word_list)
            
            # Replace numbers with numerical representation
            word_list = replace_numbers(word_list)
            
            # save tokenized document
            self.tokenized_docs.append(word_list)

            # remove stopwords andwords of 2 for fewer characters
            word_list_2 = remove_stopwords(word_list)
            word_list_2 = remove_x_chars_or_less(word_list_2, 2)

            # get term frequency (TF)
            BOW = self.get_BOW_TF(word_list_2)
            
            # get desired n_grams
            self.grams=[x for x in ngrams if x >=2 and x<=6]
            
            for size in self.grams:
                # get all ngrames of length = size
                tokenized_ngrams = ["_".join(word_list[x:x+size]) for x in range(len(word_list)) if len(word_list[x:x+size])== size]

                # remove stopwords ad correct possesive "'s" if necessary
                if config["app"]["JBRank"]["remove_stopwords"]:
                    tokenized_ngrams = remove_stopwords(tokenized_ngrams)
                    tokenized_ngrams = correct_possesives(tokenized_ngrams)
                
                # get term frequency for n grams and update original ngrams
                NBOW = self.get_BOW_TF(tokenized_ngrams)
                BOW.update(NBOW)
            
            # append this to one large Term Frequency list
            self.TF_list.append(BOW)
            
            # correct for words subsumed by ngrams
            if correct_subsum:
                self.TF_list=self.subsum_correction(TF_list)
        
        # get inverse document frequency such that idf(w) = log(total_#_of_docs / #_of_docs_with_word_'w')
        IDF_list = self.get_BOW_IDF(self.TF_list)
        
        # get tf-idf for each word, for each document
        self.tf_idf_list = self.get_TFIDF_info(tf_dicts=self.TF_list, idf_dict=IDF_list)
        
        # initialize more parameters
        self.position_cutoff=position_cutoff
        self.graph_cutoff=graph_cutoff
        self.take_top=take_top

    @staticmethod
    def title_stat_weight(w,t):
        """function for getting the title multiplier that is greater than 1 for all natural numbers, 
        monotonically increasing, and has a double dervaitve < 0 (decaying)"""
        return (w/t)*np.log(w+1)+1
        
    def stats(self):
        """further weight tf-idf by addition metrics (position of first occurrence, 
        term length, whether the word/ngram was in the title, etc)"""

        # initialize dictionary for position of first occurence factor
        PFO_factor={}

        # initialize dictionary for term length factor
        tl_factor={}

        for doc in self.tokenized_docs:
            # get the position of first occurence and term length for each term in the given document
            PFO_factor.update(self.get_PFO(doc, self.position_cutoff))
            tl_factor.update(self.get_TL(doc))

            for size in self.grams:
                # get ngrams andcorrect possesives
                tokenized_ngrams = ["_".join(doc[x:x+size]) for x in range(len(doc)) if len(doc[x:x+size])== size]
                tokenized_ngrams = correct_possesives(tokenized_ngrams)

                # get position of first occurrence and term length of each word in each document
                PFO_factor.update(self.get_PFO(tokenized_ngrams, self.position_cutoff))
                tl_factor.update(self.get_TL(tokenized_ngrams))

        # copy TF-IDF list   
        self.stat_ranking= self.tf_idf_list.copy()
        
        for i in range(len(self.stat_ranking)):
            if self.include_title:
                # tokenize document titles to get keyphrases from title
                title = self.doc_titles[i]
                title_words = word_tokenize(title)

                # get the ngrams from the title
                title_ngrams=[]
                sizes = [x for x in self.grams if x >=2 and x<=len(title_words)]
                for size in sizes:
                    title_ngrams = title_ngrams + ["_".join(title_words[x:x+size]) for x in range(len(title_words)) if len(title_words[x:x+size])== size]

                # append the title ngrams to the list of total ngrams
                title_terms = title_words+title_ngrams

            for key, val in self.stat_ranking[i].items():
                # mulitply TD-IDF score by position of first occurence factor and term length factor
                self.stat_ranking[i][key]=val*PFO_factor[key]*tl_factor[key]
                if self.include_title:
                    if key in title_terms:
                        # multiply title by decaying term > 1
                        self.stat_ranking[i][key]=self.stat_ranking[i][key]*JBRank.title_stat_weight(len(re.findall("_", key)), len(title_words))

            # rerank keyphrase scores 
            self.stat_ranking[i] = sort_dict(self.stat_ranking[i])[:self.take_top]
            self.stat_ranking[i] = {key:value for key,value in self.stat_ranking[i]}
            
    def graph(self, measure="pagerank"):
        """take top phrases and reranking with score from graphical relationship of distance to other keyphrases"""
        # initial final ranking
        self.final_rankings = {}

        # iterate over statistical rankings
        for i in range(len(self.stat_ranking)):    
            # get corresponding statistical ranking, tokenized document, and keyphrases    
            ranking = self.stat_ranking[i]
            doc = self.tokenized_docs[i]
            words = list(ranking.keys())

            # initialize networkx graph with nodes named after each keyphrase
            G=nx.Graph()
            G.add_nodes_from(words)
            
            # get edges of graphs
            # initialize dictionary to get indices of a give keyphrase
            indices_dict={}
            for word in words:
                indices_dict.update({word:self.get_inds_for_gram(word, doc)})
            
            # get the edge weights table
            weights_df=self.get_edge_weights(indices_dict, self.graph_cutoff)
            for word1 in words:
                for word2 in words:
                    # if the weight between two keyphrases is 0 or nan, do nothing
                    if weights_df[word1][word2]==0 or math.isnan(weights_df[word1][word2]):
                        pass
                    
                    # if there is a weight for the two phrases add is to the (undirected graph) and multiply it 
                    # by the ranking stat/tf-idf ranking for each phrase
                    else:
                        G.add_edge(word1, word2, weight=weights_df[word1][word2]*ranking[word1]*ranking[word2])

            if measure == "pagerank":
                # use pagerank as graph metric of importance (this is what we will use by default)
                # PageRank = 
                gr_dict=nx.pagerank(G)
            elif measure == "betweenness centrality":
                # use betweenness centrality as graph metric of importance
                # betweenness centrality = 
                gr_dict=nx.betweenness_centrality(G, weight="weight")
            elif measure == "load centrality":
                # use betweenness centrality as graph metric of importance
                # load centrality = 
                gr_dict=nx.load_centrality(G, weight="weight")
            
            # order the final rankings and export them
            gr_dict_sorted=sort_dict(gr_dict)
            self.final_rankings.update({self.doc_titles[i]: {key:value for key,value in gr_dict_sorted}})

    def get_inds_for_gram(self, word, tokenized_doc):
        """recover indices for each word and ngram"""
        # split a keyphrase into each individual word
        split_= word.split("_")
        grams=[]

        # append the indices of the given word of the ngram into a list of lists
        for wordy in split_:
            grams.append([i for i,e in enumerate(tokenized_doc) if re.search(wordy, tokenized_doc[i]) is not None])

        inds=[]
        # iterate over indices of first word in a given ngram
        for j in grams[0]:
            # store these indices in a list
            temp = [j]
            
            # iterate over all other lists of indices
            for i in range(len(grams)-1):
                # if the next word is found append the index add index and proceed to next word
                # else do not append anything
                if j+i+1 in grams[i+1]:
                    temp.append(j+i+1)
            # temp and grams are the same length, this means the word was found and we can add the index
            if len(temp)==len(grams):
                inds.append(temp)
        return inds
    
    def get_edge_weights(self, wi_dict, cutoff):
        """Use the distance between two keyphrases to get the edge weight"""
        # create data table with of the words in table to stor edge weights
        # Important: Dictionary is initialized with vaalues of nan
        edge_df=pd.DataFrame(index=list(wi_dict.keys()), columns=list(wi_dict.keys()))

        # iterate over word combinations and respective indices
        for word, indices in wi_dict.items():
            for word2, indices2 in wi_dict.items():
                # initialize distances
                dists = []

                # Do not append anything if this observation of the transpose of the matrix is
                # already filled in or the words are the same (same words implies diagonal observations)
                if math.isnan(edge_df[word2][word])==False or word==word2:
                    pass 
                else:
                    # get the length of each word (does not seem like this is used)
                    len_word=len(re.findall("_",word))
                    len_word2=len(re.findall("_",word2))

                    ngram_factor = max([len([x for x in word.split("_") if x not in word2.split("_")]), len([x for x in word2.split("_") if x not in word.split("_")])])
                    for x in wi_dict[word]:
                        dists.extend([smallest_distance(x,y)+ngram_factor for y in wi_dict[word2] if smallest_distance(x,y) < cutoff and smallest_distance(x,y)!=0])
                    num=0
                    if len(dists)==0:
                        # if the terms are not within 'cutoff' positions of each other, the weights
                        # will be 0
                        edge_df[word][word2]=0
                        edge_df[word2][word]=0
                    else:
                        # terms within the cutoff distance of each other will be given the 
                        # weight = log(cutoff/distance)
                        for dist in dists:
                            num=num+np.log(cutoff/dist)
                        edge_df[word][word2]=num/len(dists)

                        # populate other corresponding observation with 0 
                        edge_df[word2][word]=0
        return edge_df

    def get_BOW_TF(self, words):
        """returns a dictionary with the term frequency of all terms in a given document"""
        bagofwords = dict(Counter(words))
        return bagofwords

    def get_BOW_IDF(self, list_of_bows):
        """returns a dictionary with the inverse document frequency of all terms in a given corpus"""
        # get list of bag of words term frequencies
        temp = []
        for i in list_of_bows:
            temp.append(i.keys())
        
        # flatten this list of lists
        term_list = flatten(temp)
        
        # initialize an inverse document frequency dictionary
        # currently, this represents the number of documents in which the term occurs
        idf_list=dict(Counter(term_list))
        
        for key, val in idf_list.items():
            # if the term is a single word or bigram, take the log of the number of documents
            # divided by the number of documents in which the document o
            if len(re.findall("_", key))<2:
                idf_list[key] = np.log(len(list_of_bows)/val)
                
            # if the term is a 3-gram or greater, we assume it is unique to a given document
            # (mostly for the purposes of computational efficiency)
            # This differs from SGRank as SGRank considers all ngrams of n > 1 to be unique
            else:
                idf_list[key] = np.log(len(list_of_bows))
        
        return idf_list

    def get_TFIDF_info(self, tf_dicts, idf_dict):
        """takes in the term frequency dictionaries and IDF dictionary and calculates TFIDF for every term in the corpus"""
        # initialize the output dicitionary by copying the tf dict
        tf_idf_dict = tf_dicts.copy()

        # mulitply the term frequency for a given phrase in a given
        # document by the inverse document frequency of that phrase
        for i in range(0,len(tf_idf_dict)):
            for key, val in tf_idf_dict[i].items():
                tf_idf_dict[i][key] = tf_idf_dict[i][key]*idf_dict[key]
        return tf_idf_dict
    
    def get_PFO(self, tokenized_doc, cutoff_position):
        """Get the position of first occurance for a given word in a document"""
        # initialize output dictionary and possible indices
        PFO_dict={}
        nums = [j for j in range(1,len(tokenized_doc)+1)]

        for i in range(1,len(tokenized_doc)+1):
            # select i-th word from the back of the document
            word = tokenized_doc[-i]

            # if this word is in the dictionary, replace it with the current index (lower value)
            if word in PFO_dict:
                PFO_dict[word]=nums[-i]

            # else, add the word to the dictionary (with the current index)
            else:
                PFO_dict.update({word:nums[-i]})
        
        # apply the position of first occurence weighting
        for key, val in PFO_dict.items():
            PFO_dict[key] = np.log(500+(cutoff_position/val)) 
        return PFO_dict

    def get_TL(self, tokenized_doc):
        """returns the term lenght factor for a given term"""
        # whether or not you would like the term length factor to decay
        # I do want it to decay, SGRank does not do this
        decay=self.term_len_decay

        # initialize output dictionary
        tl_dict={}
        for word in tokenized_doc:
            # if decay, multiply by log of term length
            if decay:
                tl_dict.update({word:np.log(len(re.findall("_", word))+2)})
            # if not decay, multiply by term length
            else:
                tl_dict.update({word:len(re.findall("_", word))+1})
        return tl_dict

    def subsum_correction(self, tf_dict_list):
        """correct the term frequency for subsumed terms
        i.e. if two keyphrases are 'jumbo shrimp' and 'jumbo', then this would
        correct the count of 'jumbo' to not double count the word 'jumbo'
        that occurs in 'jumbo shrimp'"""

        # iterate over list of term frequency dictionaries
        for tf_dict in tf_dict_list:
            # iterate over given term frequency dictionary
            for key, val in tf_dict.items():
                # if the given term is an ngram with n>1
                if len(key.split(sep="_")) >= 2:
                    # subtract the phrase valuefrom the word if applicable
                    for word in key.split(sep="_"):
                        try:
                            tf_dict[word]=tf_dict[word]-tf_dict[key]
                        # if not pplicable then we do nothing
                        except KeyError:
                            pass
                    
        return tf_dict_list

    def run(self):
        """This is the method that the user will have to call to run the model. The TF-IDF portion is done in intitialization,
        while the statistical weighting and graphical weighting are done by the methods stats() and graph() respectively."""

        # statistical weighting
        self.stats()

        # select graph metric used for phrase importance and run final graphical weighting
        measure = config["app"]["JBRank"]["measure"]
        self.graph(measure=measure)

class SemanticAlgos(object):
    def __init__(self, docs, doc_type, sent_threshold = .3, show="Game of thrones", clean_sents = False, clean_docs = False):
        if isinstance(docs, dict) == False:
            raise TypeError("Documents must be in the form of a dictionary")
        
        cList = get_contractions()

        self.sent_threshold = sent_threshold
        self.tokenized_sents=[]
        self.doc_titles=[]
        self.doc_embeddings={}
        self.sentence_embeddings={}
        self.sentence_dists={}
        self.show=show
        self.doc_type=doc_type
        self.s3 = boto3.client("s3")
        self.bucket_name = config["aws"]["s3_bucket_name"]
        self.dynamo = boto3.resource('dynamodb')
        self.clean_sents = clean_sents
        self.clean_docs = clean_docs

        self.embed = SemanticAlgos.load_TF_Universal_Sentence_Encoder()

        for key, value in docs.items():
            title = key
            self.doc_titles.append(title)
            doc = value
            doc = doc.lower()
            
            for x in list(cList.keys()):
                doc = re.sub(x, cList[x], doc)
            
            # Replace other uncommon contractions (mainly from GOT)
            doc = re.sub(" g'", " good ", doc)
            doc = re.sub(" m'", " my ", doc)
            doc = re.sub(" d'", " do ", doc)
            doc = re.sub(" s ", "s ", doc)
            doc = re.sub("[ ]{2,}", " ", doc)

        self.cleaned_docs = docs

        # get whole document embeddings (used for both algos)
        self.get_doc_embeddings()

    @staticmethod
    def load_TF_Universal_Sentence_Encoder():
        """Load the TensorFlow Universal sentence encoder from internet or locally"""
        print("Loading TensorFlow Universal Sentence Encoder")
        embed = hub.Module(config["app"]["DAN_sentence_encoder_url"])
        return embed

    @staticmethod
    def tokenize_sentences(doc):
        """"tokenizes the sentences of a given document"""
        separators="[\.|?|!|\n|…]"
        return [x.strip() for x in re.split(separators, doc) if x.strip() != ""]

    @staticmethod
    def sentence_length_multiplier(sentence, threshold = 6):
        """Get sentence length and return our sentence weight"""
        len_sent = len(re.findall("\w \w", sentence))+1

        # If the weight is below the threshold, return nothing
        if len_sent < threshold:
            return 0
        # Else, return the sentence length multiplier
        else:
            return np.log(((len_sent-threshold)/2)+1)+1

    @staticmethod
    def get_loss(n, sim, param = .001):
        """We are looking to minimize the loss function inspired by how LASSO penalizes L2 norm.
        Loss = (1-(cosine similarity between summary and document embeddings))^2 + penalization parameter(# of sentences)"""
        return ((1-sim)**2) + param*(n)

    def get_sentence_embeddings(self, s3_path=None):
        path = config[self.show]["embeddings"][self.doc_type]["sentence_pkl_path"]
        
        if s3_path is None:
            s3_path=path

        if self.clean_sents:
            i=1
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                for title, doc in self.cleaned_docs.items():
                    self.tokenized_sents.append(SemanticAlgos.tokenize_sentences(doc))
                    if title not in list(self.sentence_embeddings.keys()):
                        print("Gathering Tensorflow Embedding")
                        sents = SemanticAlgos.tokenize_sentences(doc)
                        sent_embeddings = session.run(self.embed(sents))
                        sents_dict = {sents[i]:sent_embeddings[i] for i in range(len(sents))}
                        self.sentence_embeddings.update({title: sents_dict})
                        print("Done. You should pickle this!\n"+ str(i) +"/"+str(len(self.cleaned_docs)))
                        i = i+1
            with open(path, 'wb') as handle:
                pickle.dump(self.sentence_embeddings, handle)
            self.s3.upload_file(path, self.bucket_name, s3_path)
            os.remove(path)
        else:
            try:
                obj = self.s3.get_object(Bucket=self.bucket_name, Key=s3_path)
                self.sentence_embeddings = pickle.loads(obj['Body'].read())
            except ClientError as ex:
                if ex.response["Error"]["Code"] == 'NoSuchKey':
                    i=1
                    with tf.Session() as session:
                        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                        for title, doc in self.cleaned_docs.items():
                            self.tokenized_sents.append(SemanticAlgos.tokenize_sentences(doc))
                            if title not in list(self.sentence_embeddings.keys()):
                                print("Gathering Tensorflow Embedding")
                                sents = SemanticAlgos.tokenize_sentences(doc)
                                sent_embeddings = session.run(self.embed(sents))
                                sents_dict = {sents[i]:sent_embeddings[i] for i in range(len(sents))}
                                self.sentence_embeddings.update({title: sents_dict})
                                print("Done. You should pickle this!\n"+ str(i) +"/"+str(len(self.cleaned_docs)))
                                i = i+1
                    with open(path, 'wb') as handle:
                        pickle.dump(self.sentence_embeddings, handle)
                    self.s3.upload_file(path, self.bucket_name, s3_path)
                    os.remove(path)
                else:
                    raise ex

    def get_doc_embeddings(self):
        self.doc_embeddings=aws.get_dynamo_data(item=config[self.show]["embeddings"][self.doc_type]["doc_data_name"], 
        table=config["aws"]["Dynamo_Table"], resource=self.dynamo, embedding=True)

        if self.doc_embeddings=="No Response" or self.clean_docs:
            print("Generating New Document Embeddings")
            self.doc_embeddings={}
            i=1
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                for title, doc in self.cleaned_docs.items():
                    if title not in list(self.doc_embeddings.keys()):
                        print("Gathering Tensorflow Embedding")
                        embeddings = session.run(self.embed([doc]))
                        self.doc_embeddings.update({title: embeddings[0]})
                        print("Done. You should pickle this!\n"+ str(i) +"/"+str(len(self.cleaned_docs)))
                        i = i+1

            dynamo_embeddings = deepcopy(self.doc_embeddings)
            dynamo_embeddings = aws.handle_numpy(dynamo_embeddings)
            dynamo_embeddings.update({config["aws"]["partition_key"]:config[self.show]["embeddings"][self.doc_type]["doc_data_name"]})
            dynamo_table = self.dynamo.Table(config["aws"]["Dynamo_Table"])
            dyname_table.put_item(Item=dynamo_embeddings)

    def graph_text_summarization(self, max_sents=config["app"]["text_summarization"]["max_num_sentences"], min_sents=config["app"]["text_summarization"]["min_num_sentences"], measure=config["app"]["text_summarization"]["measure"], order_by_occurence=config["app"]["text_summarization"]["order_by_occurence"]):
        """Text Summarization Algorithm that I wrote based on LexRank paper. More info will follow in README"""
        item = config[self.show]["text_summ_name"][self.doc_type]
        results = aws.get_dynamo_data(item=item, table=config["aws"]["Dynamo_Table"], resource=self.dynamo)
        # print(results)
        if results == "No Response" or self.clean_sents:
            partition_key = config["aws"]["partition_key"]
            results = {partition_key:item}
            self.get_sentence_embeddings()
            for title, sent_embed in self.sentence_embeddings.items():
                # do graph stuff
                dists = 1-pairwise_distances(list(sent_embed.values()), metric="cosine")
                dists = pd.DataFrame(dists, index = list(sent_embed.keys()), columns=list(sent_embed.keys()))
                G = nx.Graph()
                G.add_nodes_from(list(dists.columns))
                for i in list(dists.columns):
                    for j in list(dists.index):
                        if i==j or math.isnan(dists[i][j]) or dists[i][j] < self.sent_threshold:
                            pass
                        else:
                            G.add_edge(i, j, weight=dists[i][j])
                            dists[j][i] = None
                if measure == "pagerank":
                    gr_dict = nx.pagerank(G)
                elif measure == "betweenness centrality":
                    gr_dict = nx.betweenness_centrality(G, weight="weight")
                elif measure == "load centrality":
                    gr_dict = nx.load_centrality(G, weight="weight")

                for key, value in gr_dict.items():
                    gr_dict[key] = value*SemanticAlgos.sentence_length_multiplier(key)

                temp_dict = sort_dict(gr_dict)

                doc_embed=self.doc_embeddings[title]
                sorted_gr_dict = temp_dict[0:max_sents]
                summ_embedding_dict={}

                # finding optimal length for each summarization
                print("Finding Optimal n...")
                for i in range(min_sents, max_sents+1):
                    test_dict = dict(sorted_gr_dict[0:i])
                    # print(list(self.sentence_embeddings.items()))
                    test_dict = {key:self.sentence_embeddings[title][key] for key in list(test_dict.keys())}
                    summ_embedding = np.mean(np.array(list(test_dict.values())), axis = 0)
                    summ_embedding_dict.update({str(i):summ_embedding})
                        
                embed_list = [doc_embed]+list(summ_embedding_dict.values())
                dists = 1-pairwise_distances(embed_list, metric="cosine")[0]
                sim_dict = dict(zip(range(min_sents, max_sents+1), dists[1:len(dists)]))
                loss = [SemanticAlgos.get_loss(key, val) for key, val in sim_dict.items()]
                n = list(sim_dict.keys())[loss.index(min(loss))]
                if isinstance(n, list):
                    n = n[0]
                print("Optimal Number of Sentences: " + str(n))
                sorted_gr_dict = dict(temp_dict[0:n])
                if order_by_occurence:
                    summary_dict = {x:gr_dict[x] for x in list(gr_dict.keys()) if x in list(sorted_gr_dict.keys())}
                else:
                    summary_dict = sorted_gr_dict
                results.update({title:list(summary_dict.keys())})
            dynamo_table = self.dynamo.Table(config["aws"]["Dynamo_Table"])
            dynamo_table.put_item(Item=results)
        return(results)

    def text_similarity(self, take_top=config["app"]["text_similarity"]["take_top"]):
        results={}
        doc_titles = list(self.doc_embeddings.keys())
        doc_embeddings = list(self.doc_embeddings.values())
        dists=1-pairwise_distances(doc_embeddings, metric="cosine")
        for i in range(len(dists)):
            similarity_dict={}
            for j in range(len(dists)):
                if i == j:
                    pass
                else:
                    similarity_dict.update({doc_titles[j]: dists[i][j]})
            sorted_doc_dict = sort_dict(similarity_dict)[0:take_top]
            results.update({doc_titles[i]: {key:value for key,value in sorted_doc_dict}})
        # print(results)
        # temp_results=sort_dict(results)
        # final_results=temp_results[0:take_top]
        # final_results= {key:value for key,value in final_results}
        return(results)

if __name__ == "__main__":
    show = "Game of thrones"
    pick = config["app"]["use_s3"]
    episodes, season_dict, data = process_episodes(show, S3=pick)
    ep_algs = SemanticAlgos(episodes, doc_type="episodes", sent_threshold=config["app"]["text_summarization"]["sentence_similarity_threshold"], show=show)
    summs = ep_algs.graph_text_summarization()
    for key, val in summs.items():
        df = data[data.episode == key][["character_name", "narration"]]
        new_val = add_ep_speakers(df, val)
        summs[key] = new_val
    print(summs)