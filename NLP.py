# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 01:29:33 2018

@author: Joe
"""

"""Use My Own Algorithm to Extract Keyphrases"""
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import networkx as nx
import pickle
import pandas as pd 
import inflect
from collections import Counter
import re
import numpy as np
from Scraper import Genius_TV_Scraper
import math
import sys

# return second element of tuple within a list
def second(elem):
    """takes second item of list"""
    return elem[1]

# flatten a list of lists
def flatten(x):
    """Flattens a list of lists into a single list"""
    flattened = [val for sublist in x for val in sublist]
    return flattened

def sort_dict(dict_unsorted):
    dict_sorted = sorted(dict_unsorted.items(), key=lambda x: x[1], reverse=True)
    return dict_sorted

def get_inds_for_gram(word, tokenized_doc):
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

def smallest_distance(list1, list2):
    a = 0
    b = 0
    distance = sys.maxsize 
    while (a < len(list1) and b < len(list2)): 
        if (abs(list1[a] - list2[b]) < distance): 
            distance = abs(list1[a] - list2[b]) 
        if (list1[a] < list2[b]): 
            a += 1
        else: 
            b += 1 
    return distance

def get_edge_weights(wi_dict, cutoff):
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

def pickle_data_frames():
    """Scrapes Genius scripts and saves them as Pickle files to be accessed later and to be easily called by API."""
    GOT_Scraper = Genius_TV_Scraper(show='Game of thrones')
    GOT_data = GOT_Scraper.get_scripts()
    GOT_data.to_pickle('GOT_Pickle.pkl')
    Office_Scraper = Genius_TV_Scraper(show='The office us')
    Office_data = Office_Scraper.get_scripts()
    Office_data.to_pickle('Office_Pickle.pkl')

def process_episodes(show, seasons=None):

    # initialize scraper for desire show and seasons (default for seasons is all available on Genius)
    scraper = Genius_TV_Scraper(show=show, seasons=seasons)

    #scrape previously specified show and seasons
    data = scraper.get_scripts()

    #get episodes from data
    z = [sorted(data["Episode"], key=data["Episode"].count, reverse=True)[0]]
    for i in sorted(data["Episode"], key=data["Episode"].count, reverse=True):
        if i not in z:
            z.append(i)
    
    docs=[]
    for episode in z:
        df = data[data.Episode == episode]
        doc = " ".join(list(df["Line"]))
        docs = docs.append(doc)

    return docs

def process_characters(show, seasons=None, num_char=10):

    # initialize scraper for desire show and seasons (default for seasons is all available on Genius)
    scraper = Genius_TV_Scraper(show=show, seasons=seasons)

    # scrape previously specified show and seasons
    data = scraper.get_scripts()

    z = [sorted(data["Character_Name"], key=data["Character_Name"].count, reverse=True)[0]]
    for i in sorted(data["Character_Name"], key=data["Character_Name"].count, reverse=True):
        if i not in z:
            z.append(i)

    for char in z[0:num_char]:
        df = data[data.Character_Name == char]
        char_text = " ".join(list(df["Line"]))
        doc = textacy.Doc(char_text, lang=u'en')
        docs.append(doc)
    corpus = textacy.Corpus(lang=u'en', docs=docs)
    idf_dict = corpus.word_doc_freqs(normalize=None, weighting="idf", as_strings=True)

    return docs, idf_dict

def clean_text(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Removes English Stopwords"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    rep_num = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = rep_num.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def get_BOW_TF(words):
    """returns a dictionary with the term frequency of all terms in a given document"""
    bagofwords = dict(Counter(words))
    #Sum = sum(bagofwords.values())
    
#    for key, val in bagofwords.items():
#        bagofwords[key] = val
    return bagofwords

def get_BOW_IDF(list_of_bows):
    """returns a dictionary with the inverse document frequency of all terms in a given corpus"""
    temp = []
    for i in list_of_bows:
        temp.append(i.keys())
    
    term_list = flatten(temp)
    
    idf_list=dict(Counter(term_list))
    
    for key, val in idf_list.items():
        if len(re.findall("_", key))==0:
            idf_list[key] = np.log(len(list_of_bows)/val)
            
        else:
            idf_list[key] = np.log(len(list_of_bows))
    
    return idf_list

def get_TFIDF_info(tf_dicts, idf_dict):
    """takes in the term frequency dictionaries and IDF dictionary anc calculates TFIDF for every term in the corpus"""
    tf_idf_dict = tf_dicts.copy()
    for i in range(0,len(tf_idf_dict)):
        for key, val in tf_idf_dict[i].items():
            tf_idf_dict[i][key] = tf_idf_dict[i][key]*idf_dict[key]
#    list1 = [x.items() for x in tf_idf_dict]
#    foo = [list(x) for x in list1]
#    for i in range(0,len(foo)):
#        foo[i].sort(key=second, reverse=True)
#    return foo
    return tf_idf_dict

def get_PFO(tokenized_doc, cutoff_position):
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

def get_TL(tokenized_doc):
    tl_dict={}
    for word in tokenized_doc:
        tl_dict.update({word:np.log(len(re.findall("_", word))+2)})
        
    return tl_dict

def subsum_correction(tf_dict_list):
    for tf_dict in tf_dict_list:
        for key, val in tf_dict.items():
            if len(key.split(sep="_")) >= 2:
                for word in key.split(sep="_"):
                    tf_dict[word]=tf_dict[word]-tf_dict[key]
                
    return tf_dict_list
        
class MyAlgo(object):
    def __init__(self, docs, ngrams=[1,2,3,4,5,6], cutoff=1000, take_top=50):
        cList = {
                "ain't": "am not",
                "aren't": "are not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "couldn't've": "could not have",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "em'": "them",
                "'em": "them",
                "hadn't": "had not",
                "hadn't've": "had not have",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'd've": "he would have",
                "he'll": "he will",
                "he'll've": "he will have",
                "he's": "he is",
                "how'd": "how did",
                "how'd'y": "how do you",
                "how'll": "how will",
                "how's": "how is",
                "i'd": "i would",
                "i'd've": "i would have",
                "i'll": "i will",
                "i'll've": "i will have",
                "i'm": "i am",
                "i've": "i have",
                "isn't": "is not",
                "it'd": "it had",
                "it'd've": "it would have",
                "it'll": "it will",
                "it'll've": "it will have",
                "it's": "it is",
                "let's": "let us",
                "ma'am": "madam",
                "mayn't": "may not",
                "might've": "might have",
                "mightn't": "might not",
                "mightn't've": "might not have",
                "must've": "must have",
                "mustn't": "must not",
                "mustn't've": "must not have",
                "needn't": "need not",
                "needn't've": "need not have",
                "o'clock": "of the clock",
                "oughtn't": "ought not",
                "oughtn't've": "ought not have",
                "shan't": "shall not",
                "sha'n't": "shall not",
                "shan't've": "shall not have",
                "she'd": "she would",
                "she'd've": "she would have",
                "she'll": "she will",
                "she'll've": "she will have",
                "she's": "she is",
                "should've": "should have",
                "shouldn't": "should not",
                "shouldn't've": "should not have",
                "so've": "so have",
                "so's": "so is",
                "that'd": "that would",
                "that'd've": "that would have",
                "that's": "that is",
                "there'd": "there had",
                "there'd've": "there would have",
                "there's": "there is",
                "they'd": "they would",
                "they'd've": "they would have",
                "they'll": "they will",
                "they'll've": "they will have",
                "they're": "they are",
                "they've": "they have",
                "to've": "to have",
                "wasn't": "was not",
                "we'd": "we had",
                "we'd've": "we would have",
                "we'll": "we will",
                "we'll've": "we will have",
                "we're": "we are",
                "we've": "we have",
                "weren't": "were not",
                "what'll": "what will",
                "what'll've": "what will have",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "when's": "when is",
                "when've": "when have",
                "where'd": "where did",
                "where's": "where is",
                "where've": "where have",
                "who'd": "who had",
                "who'll": "who will",
                "who'll've": "who will have",
                "who's": "who is",
                "who've": "who have",
                "why's": "why is",
                "why've": "why have",
                "will've": "will have",
                "won't": "will not",
                "won't've": "will not have",
                "would've": "would have",
                "wouldn't": "would not",
                "wouldn't've": "would not have",
                "y'all": "you all",
                "y'alls": "you alls",
                "y'all'd": "you all would",
                "y'all'd've": "you all would have",
                "y'all're": "you all are",
                "y'all've": "you all have",
                "you'd": "you had",
                "you'd've": "you would have",
                "you'll": "you wil",
                "you'll've": "you will have",
                "you're": "you are",
                "you've": "you have",
                "m'lord": "my lord",
                "m'lady": "my lady",
                "d'you": "do you"
                } 
        
        self.TF_list=[]
        self.tokenized_docs=[]
    
        for doc in docs:
            # Handle contractions
            #CODE
            doc = doc.lower()
            
            for x in list(cList.keys()):
                doc = re.sub(x, cList[x], doc)
            
            # Replace other uncommon contractions (mainly from GOT)
            doc = re.sub("g'", "good ", doc)
            doc = re.sub("m'", "my ", doc)
            doc = re.sub("d'", "do ", doc)
            doc = re.sub("'s", " is", doc)
            
            # Tokenize document
            word_list = word_tokenize(doc)
            
            # Clean Text
            word_list = clean_text(word_list)
            
            # Replace numbers with numerical representation
            word_list = replace_numbers(word_list)
            
            self.tokenized_docs.append(word_list)
            
            BOW = get_BOW_TF(word_list)
            
            self.grams=[x for x in ngrams if x >=2 and x<=6]
            
            for size in self.grams:
                tokenized_ngrams = ["_".join(word_list[x:x+size]) for x in range(len(word_list)) if len(word_list[x:x+size])== size]
                NBOW = get_BOW_TF(tokenized_ngrams)
                BOW.update(NBOW)
            
            self.TF_list.append(BOW)
            
            #self.TF_list=subsum_correction(TF_list)
            
        IDF_list = get_BOW_IDF(self.TF_list)
        
        self.tf_idf_list = get_TFIDF_info(tf_dicts=self.TF_list, idf_dict=IDF_list)
        
        self.cutoff=cutoff
        self.take_top=take_top
        
    def stats(self):
        PFO_factor={}
        tl_factor={}
        for doc in self.tokenized_docs:
            PFO_factor.update(get_PFO(doc, self.cutoff))
            tl_factor.update(get_TL(doc))
            for size in self.grams:
                tokenized_ngrams = ["_".join(doc[x:x+size]) for x in range(len(doc)) if len(doc[x:x+size])== size]
                PFO_factor.update(get_PFO(tokenized_ngrams, self.cutoff))
                tl_factor.update(get_TL(tokenized_ngrams))
                
        self.stat_ranking= self.tf_idf_list.copy()
        
        for i in range(len(self.stat_ranking)):
            for key, val in self.stat_ranking[i].items():
                self.stat_ranking[i][key]=val*PFO_factor[key]*tl_factor[key]
            self.stat_ranking[i] = sort_dict(self.stat_ranking[i])[:self.take_top]
            self.stat_ranking[i] = {key:value for key,value in self.stat_ranking[i]}
            
    def graph(self):
        self.pagerank_list = []
        for i in range(len(self.stat_ranking)):        
            ranking = self.stat_ranking[i]
            doc = self.tokenized_docs[i]
            words = list(ranking.keys())
            G=nx.Graph()
            G.add_nodes_from(words)
            
            #get edges of graphs
            indices_dict={}
            for word in words:
                indices_dict.update({word:get_inds_for_gram(word, doc)})

            weights_df=get_edge_weights(indices_dict, self.cutoff)
            for word1 in words:
                for word2 in words:
                    if weights_df[word1][word2]==0 or math.isnan(weights_df[word1][word2]):
                        pass
                    else:
                        G.add_edge(word1, word2, weight=weights_df[word1][word2]*ranking[word1]*ranking[word2])
            pr=nx.pagerank(G) 
            pr_sorted=sort_dict(pr)
            self.pagerank_list.append({key:value for key,value in pr_sorted})
        for i in self.pagerank_list:
            print(i)
            print("*"*30)

if __name__ == "__main__":
    with open('examples.pkl', 'rb') as f:
        docs = pickle.load(f)
    x=MyAlgo(docs=docs, ngrams=[1,2])
    x.stats()
    x.graph()