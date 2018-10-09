# NIPS 
Topic modeling of NIPS 2015 papers using FastTex word embeddings  

Summary of methodology: 
FastTex is trained on a corpus of NIPS 2015 papers extracted from https://www.kaggle.com/benhamner/nips-2015-papers/version/2/home
I run two separate clusterings: 1. on all words 2. on calculated embeddings of papers. While the latter gives an straight forward answer to the problem of grouping related papers, the first methodology outputs relevant keywords that overall represent the content of these papers. I also have used the first methodology to identify what fraction of each paper is about each word-cluster.

- Standard text preprocessing steps such as stopwords removal, part-of-speech tagging and lemmatization are implemented and the user can turn on/off those options in the code.
- The result of preprocessing is saved in a .csv file.
- Bigrams and Trigrams of the corpus are made.
- Gensim FastTex implementation is used to train word embeddings. The trained model is saved in a .txt file.
- H2O machine learning platform is used for Kmeans clustering on word embeddings.
- Top 10 words in each cluster is shown in a word cloud and is saved in a .csv file.
- The share of each word-cluster in calculated for each paper.
- PCA is used to estimate papers embedding vectors.
- Another Kmeans clustering is performed on papers and the results are saved in a .csv file.
- 2D TSNE digram is used for visualizing paper clusters.

In summary, .py file can be run as following:

Example:  python NIPS-IBM.py '/Users/mlamee/Downloads/IBM/output/Papers.csv' 1 1 5 1000 > log.txt
 - log.txt contains all the terminal outputs of the run.
 - The first argument is the filename of the main data file.
 - The 2nd argument is a boolen variable (1/0) that determines if stopwords should be removed (default=1).
 - The 3rd argument is a boolen variable (1/0) that determines if words should be lemmatized (default=1).
 - The 4th argument is an integer and determines the window size of the FastTex embeddings (default=5).
 - The 5th argument is an integer and determines the size of embedding vectors (default=1000).

Overall, along the code, there are many functions with many parameters that can be set/modified. However, in the main function I decided to only have the option for a handful of them. 
