# TV_NLP_Project
##Natural Language Processing Algorithms
###JBRank
JBRank is my attempt to replicate and improve upon the SGRank algorithm for keyphrase extraction from the University of Colorado. I thought that writing a machine learning algorithm from scratch like this would be a great exercise, as well as making adjustments where I thought they were necessary. 
####Step 1
The first step of JBRank is extracting all words and n-grams each document. SGRank generates all 1 through 6 grams for a given document, so I made JBRank default to do the same. That being said, I am only using 1, 2, and 3 grams for the purpose of the web app. In SGRank, uncommon ngrams for a particular documents are eliminated depending on length. I find this part to be useful, but a bit cumbersome for an initial gathering of keyphrases. While JBRank currently has no measure like this, it is something I would may like to impliment in the future. One feature of SGRank the JBRank does currently share is the removal of ngrams with stopwords in them. However, I have only removed keyphrases that have stopwords at either the beginning or end. For example, since "of" is a stopword phrases like "Lord of Light" will not be eliminated, but phrases "King of" will be. 
####Step 2
JBRank, like SGRank, also uses TF-IDF weighting to get its initial rankings for the phrases, but it does this in a slightly different way. The TF-IDF calculation is done by taking the raw frequency of each phrase in a particular document (as oppsed to the relative frequency) and multiplying that by the inverse document frequency (ln(number_of_total_documents/number_of_documents_with_term)). A key difference here is that SGRank assumes that all n-grams where n>1 are unique to its particular document. I got confusing results when I implimented this for my Game of Thrones scripts, even though the authors of SGRank decided to consider all n-grams to be document unique, due to accuracy issues. Therefore I decided to calculate a true TF-IDF for all 1 and 2 grams, while assuming all keyphrases of 3 words or greater to be unique. Another key difference between JBRank and SGRank is that the latter corrects for subsumed words, while the former does not. This means this JBRank counts the frequency of a phrase, even if it is included within anohter phrase. This is because the statistical ranking system that JBRank uses favors phrases of more words, so not correcting this allows shorter phrases to not have their importance skewed downward. I wrote a method of the JBRank class to complete this correction, but have the option to not use it as the default.
####Step 3
The next step of JBRank is to use additional metrics to generate the top X key phrases per document (default value is 50 for JBRank, 100 for SGRank). The two other metrics considered for each phrase are the position of first occurance in a given document and the term length. In regards to position, a linear decay function is used to favor terms that occur earlier in a body of text. There is also a cutoff position, meaning that words that first occur after this position are no longer considered. This position defaults to the 3000th word in SGRank, but JBRank defaults to 5000 as the cutoff. Even after this change, the JBRank still seemed to disproportionally favor terms that occur earlier. While this makes sense for academic papers with an abstract, television scripts are not formetted like this. Therefore, I decidec to dampen this effect. The position of first occurence factor is (ln(500+(cutoff_position/position_of_first_occurence)). Adding 500 in this equation greatly reduces the variance in values for this factor, allowing it to effet the ranking less. Adding 500 in this equation is unique to JBRank.
The raw term length is also considered in SGRank, but this resulted in excessive favoring of longer terms. Therefore, JBRank uses ln(term_length). The final statistical ranking is tf_idf_factor * position_factor * term_length factor.
####Step 4
The final step of the JBRank algorithm is reranking the top 50 phrases using graph metrics. We use the networkx module in python to contruct weighted graphs for each document where each node corresponds to one of the top 50 keyphrases. First, the weights between any two given terms. These weights are (score_of_term_i * score_of_j * average_dictance_factor). Using the default arguments, the average distance factor considers all times the terms occur within at least 500 positions of eachother in the document. Here 500 is the window size, but this is a default of 1500 for SGRank. This average distance factor is the sum of ln(window_size/distance of terms i and j)/number_of_term_cooccurances for each instance for terms i and j in a particular document. 
Once all of the graphs are generated, the pagerank is calculated for each node in each graph. (Will explain Larry Page's pagerank in future). These pagerank values are now the final rankings for each of the keyphrases
###Text Summarization

##References
Danesh, Soheil, et al. “SGRank: Combining Statistical and Graphical Methods to Improve the State of the Art in Unsupervised Keyphrase Extraction.” Aclweb.org, 5 June 2015, www.aclweb.org/anthology/S15-1013.
Daniel, et al. “Universal Sentence Encoder.” ArXiv.org, Google Research, 12 Apr. 2018, arxiv.org/abs/1803.11175.
Erkan, Güneş, and Dragomir R Radev. “LexRank: Graph-Based Lexical Centrality as Salience in Text Summarization.” LexRank: Graph-Based Lexical Centrality as Salience in Text Summarization, www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html.
Iyyer, Mohit, et al. “Deep Unordered Composition Rivals Syntactic Methods for Text Classification.” Deep Unordered Composition Rivals Syntactic Methods for Text Classification, people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf.
Page, Larry. “The PageRank Citation Ranking: Bringing Order to the Web.” The PageRank Citation Ranking: Bringing Order to the Web, 1999, ilpubs.stanford.edu:8090/422/1/1999-66.pdf.