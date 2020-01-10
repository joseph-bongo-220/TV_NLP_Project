import re

def get_inds_for_gram(word, tokenized_doc):
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