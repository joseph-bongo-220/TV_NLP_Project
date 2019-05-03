import re
import sys
import pandas as pd
import math
import numpy as np

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
                print(word+"\n"+word2+"\n"+str(ngram_factor))
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
