
# coding: utf-8

# ### Data challenge for IBM
# Please email your questions to mehdi.lamee@gmail.com
# Thanks.

# In[1]:

import sys
import time
import string
import pandas as pd
import numpy as np
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()

pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',2000)
pd.set_option('display.max_colwidth',-1)
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.float_format', lambda x: '%.4f' % x)

import matplotlib.pyplot as plt
#%matplotlib inline 
import seaborn as sns
sns.set(color_codes=True)
#%config InlineBackend.figure_format = 'retina'

stemmer = nltk.stem.porter.PorterStemmer()
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from stop_words import get_stop_words
import math as math
import imp
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator

from wordcloud import WordCloud
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import FastText
# spacy for lemmatization
#import spacy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings(action='once')


# In[37]:


# Parsing text and keeping .!? for identifying sentenses.
def parse(s):
    temp = re.sub(r'[^A-Za-z.!?\s]', r' ', str(s)).strip().lower()
    return(temp)

# Parsing text and only keeping alphabetical charecters 
def parse2(s):
    temp = re.sub(r'[^A-Za-z\s]', r' ', str(s)).strip()
    return(temp)

# Identify and group high level adjectives, verbs, nouns, etc. 
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement 
    
# Lemmatization of words based on the their part of speech tag    
def lemma(x):
    wntag = get_wordnet_pos(x['pos'])
    if wntag is None:# not supply tag in case of None
        lemma = wordnet_lemmatizer.lemmatize(x['word']) 
    else:
        lemma = wordnet_lemmatizer.lemmatize(x['word'], pos=wntag)
    return lemma

# Text preprocessing function. It accepts the data file and has multiple options: \
#   1. remove stopping words, 2. Adding a list of manual stopwords, 3. Part of speech tagging, 4. Keeping only a set of tags: e.g. nouns and adjectives, 
#  5. Lemmatization. All of these options can be turned on and off
def prep(data,lemmatize=True,remove_stops=True,manuals=[],keeplist=[]):
    msh=pd.DataFrame()
    start_time = time.time()
    #print start_time
    # Iterating over text of each dataer
    num_docs=data.shape[0]
    for i in range(num_docs):
        print 'Pre-processing document ', i+1, ' of ', num_docs
        # Parse the text and only keep alphabetical charecters. Then tokenize and run a part of speech tagger. 
        tmp=pd.DataFrame(nltk.pos_tag(nltk.word_tokenize(parse(data.iloc[i,-1])),lang='eng'))
        
        # Adding an if statement to check if the tmp dataframe actually has values to avoid errors due to problems in data. (One of the article's text is just garbage.)
        if tmp.shape[0]>0:
            if len(keeplist)>0:
                # Only keep words with specified part of speech tags
                tmp=tmp[tmp[1].isin(keeplist)]
            tmp['paper_id']=data.iloc[i,0]
            msh=msh.append(tmp)
    # Renaming column names to something that make sense    
    msh=msh.rename(columns={0:'word',1:'pos'})        
    if lemmatize==1:
        print 'Lemmatization of words is activated...'
        msh['word']=msh.apply(lemma, axis=1)
        
    print msh.columns
    if remove_stops==1:
        print 'Removing stopwords is activated...'
        stopwords = pd.DataFrame({'stop_words':np.array(list(nltk.corpus.stopwords.words('english')))})
        if len(manuals)>0:
            manual_stops=pd.DataFrame({'stop_words':manuals})
            alphabet=pd.DataFrame({'stop_words':np.array(list(string.ascii_lowercase))})
            manual_stops=manual_stops.append(alphabet)
            manual_stops=manual_stops.append(alphabet+'.')
            print 'manual stopwords list: ', manual_stops
        else:         
            manual_stops=pd.DataFrame({'stop_words':np.array(list(string.ascii_lowercase))})
        stopwords=stopwords.append(manual_stops)
        #Getting rid of stopping words
        msh=msh[~msh['word'].isin(stopwords.values[:,0])]

    # Getting rid any potential NaN value
    msh=msh[~msh.word.isnull()]
    
    # Saving the processed data
    print 'Saving the preprocessed data to "msh.csv"'
    msh.to_csv('msh.csv',index=False)
    print("--- %s minutes ---" % (time.time()/60.0 - start_time/60.0))
    return msh

# Training a FastText embedding model with size=1000 for each word vector. I also set the window parameter to 5 and minimum counts of each word to be considered 3.
def train_fasttex(data,min_count=3,window=5,size=1000,name='model.txt'):
    start_time = time.time()
    msh_fastex = FastText(data, min_count=min_count, workers=8,window=window,size=size)
    print 'Saving the FastTex model in ',name
    msh_fastex.wv.save_word2vec_format(name, binary=False)
    print("--- %s minutes ---" % (time.time()/60.0 - start_time/60.0))
    return msh_fastex

# Evaluating the results of K-means clustering models by calculating metrics such as AIC and BIC 
def diagnostics_from_clusteringmodel(model):
    total_within_sumofsquares = model.tot_withinss()
    number_of_clusters = len(model.centers())
    number_of_dimensions = len(model.centers()[0])
    number_of_rows = sum(model.size())
    
    aic = total_within_sumofsquares + 2 * number_of_dimensions * number_of_clusters
    bic = total_within_sumofsquares + math.log(number_of_rows) * number_of_dimensions * number_of_clusters
    
    return {'Clusters':number_of_clusters,
            'Total Within SS':total_within_sumofsquares, 
            'AIC':aic, 
            'BIC':bic}

# Accepts a dataframe with a least two columns: 'word' list and its associated topic cluster: 'Cluster'
# Outputs a word cloud diagram of each topic with first 20 words in each cluster as well as a new dataframe with freqquency-weights of each word in each cluster. 
def cloud_plot(data, corpus, name):
    num=data.Cluster.nunique()
    sqt=int(np.round(math.sqrt(num)))
    print 'SQT ',sqt
    print 'Num ',num
    if (sqt**2>=num):
        row=sqt
        col=sqt
    else:
        row=sqt+1
        col=sqt
    print 'row ', row, ' col ', col
    plt.figure(1)
    ax,fig= plt.subplots(row,col,dpi=75,figsize=(col*10,row*8),tight_layout=True)
    plt.axis("off")
    i=1
    weights=pd.DataFrame({'word':[]})
    for group in data.Cluster.unique():
        plt.subplot(row,col,i)
        #make a temporary  dataframe with all words in cluster=group
        flat_corpus = pd.Series([item for sublist in corpus for item in sublist])
        tmp=flat_corpus[flat_corpus.isin(data[data.Cluster== group].word.values)].values
        # Calculates weights of each word in each cluster
        weights = pd.concat([weights,pd.DataFrame(1.0*pd.DataFrame({'word':tmp}).word.value_counts()/tmp.shape[0])])
        text=' '.join(tmp)
        wordcloud = WordCloud(max_words=10,max_font_size=40,collocations=False,repeat=False,prefer_horizontal=1,width=400, height=400, background_color="whitesmoke",color_func=lambda *args, **kwargs: "Teal").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title('Cluster '+str(group),fontsize=30)
        plt.margins(x=0, y=0)
        i=i+1
    weights=weights.rename(columns={'word':'weight'},index=str)
    data=data.set_index('word').join(weights,how='left')
    plt.savefig(name,format='pdf',facecolor='white', edgecolor='none')
    #plt.show()
    return data

# Generates a TSNE plot of the input document. witn n_components=n number of PCA components and the assigned perplexity.
def tsne_plot(docs,name,perplexity=3, n_components=2):
    "Creates and TSNE model and plots it"
    labels = docs.iloc[:,0]
    tokens = []

    for i in range(docs.shape[0]):
        tokens.append(docs.iloc[i,1:])
        #labels.append(word)
    
    tsne_model = TSNE(perplexity=perplexity, n_components=n_components, init='pca', n_iter=2500, random_state=230)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(20, 20)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(name,format='pdf')
    #plt.show()

# In[2]:

# I define the main function so we can run this from the terminal and assign the argument values for some of the variables
# The first argument: filename is the main data file
# The 2nd argument is a boolen variable (1/0) that determines if stopwords should be removed (default=1)
# The 3rd argument is a boolen variable (1/0) that determines if words should be lemmatized (default=1)
# The 4th argument is an integer and determines the window size of the FastTex embeddings (default=5)
# The 5th argument is an integer and determines the size of embedding vectors (default=1000)

# Example:  python NIPS-IBM.py '/Users/mlamee/Downloads/IBM/output/Papers.csv' 1 1 5 1000 > log.txt

def main():
    script = sys.argv[0]
    filename = sys.argv[1]
    remove_stops=int(sys.argv[2])
    lemmatize=int(sys.argv[3])
    window=int(sys.argv[4])
    vectorsize=int(sys.argv[5])

    # reading the main data file
    pap=pd.read_csv(filename)
    print 'Shape of data file: ', pap.shape
    print 'Column titles: ', pap.columns
    #pap.head(1)


    # In[4]:


    # Adding some manual stop words
    manuals=['et', 'al','page']
    # Only keeping Nouns, adjectives and verbs
    keeplist=['NNS', 'VBP', 'VBN', 'NN', 'VBD', 'VBZ','VBG', 'JJ', 'VB', 'JJR', 'JJS', 'NNP','NNPS']
    msh=prep(pap,lemmatize=lemmatize,remove_stops=remove_stops,manuals=manuals,keeplist=keeplist)
    #print msh.shape
    #msh.head()


    # In[5]:


    # Making a list corpus of tokenized papers
    print 'Making a corpus of the tokenized papers'
    papers=[]
    for i in msh.paper_id.unique():
        papers=papers+[list(msh[msh.paper_id==i].word)]


    # In[6]:

    # Make bigrams of words
    def make_bigrams(texts,bigram_mod):
        return [bigram_mod[doc] for doc in texts]

    # Make trigrams of words
    def make_trigrams(texts,bigram_mod,trigram_mod):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
        # Build the bigram and trigram models
    bigram = gensim.models.Phrases(papers, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[papers], threshold=100)  

    #trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # making Tri-bigrams
    print 'Making Trigrams of the corpus'
    papers_trigrams = make_trigrams(make_bigrams(papers,bigram_mod=bigram_mod),bigram_mod=bigram_mod,trigram_mod=trigram_mod)

    # Just making a replica array of papers_trigrams but with the paper_ids
    j=0
    paper_id=[]
    for i in msh.paper_id.unique():
        paper_id=paper_id+[[i]*len(papers_trigrams[j])]
        j=j+1


    # ### Training a FastTex embedding on our Tri-gram version of corpus.
    # I choose size of 1000, window width of 5 and minimum word count of 3. These choices are not optimized.   
    # The result will be saved in a text file provided in "name" variable.

    # In[7]:

    print 'Training the FastTex embeddings and saving the result in "nips_fasttex.txt"'
    print 'window = ', window, 'vectorsize= ', vectorsize
    msh_fastex=train_fasttex(papers_trigrams,min_count=3,window=window,size=vectorsize, name='nips_fasttex.txt')


    # ### Preparing data
    # At this stage, I will use the fast implementation of a K-means clustering algorithm in H2O machine learning platform
    # to cluster all extracted words in our corpus based on their trained embedings. These clusters can be used for multiple purposes:
    # 1. The top most frequent keywords in each cluster can be used as potential important keywords related to the context of papers and field. 
    # In other words, using this techique, we should be able to extract most relevant keywords of the corpus that represent the NIPS conference. We also will know the context of each cluster of keywords. 
    # One advatage of this methodology is that not only we will capture the dominant and most frequent sets of keywords, we also will identify small subsets of keywords related to a technical field that only a few papers have talked about. 
    # 2. We later will use these word cluster as a proxy to probability of belonging each paper to each of these clusters.

    # In[9]:


    h2o.init()


    # In[10]:

    print 'Converting data to H2O dataframe.'
    X = msh_fastex[msh_fastex.wv.vocab]
    vocabs=np.array(list(msh_fastex.wv.vocab))
    result=pd.DataFrame(X)
    result['word']=vocabs
    result=result.set_index('word')
    newwd_h=h2o.H2OFrame(result.reset_index(),column_names=list(result.reset_index().columns.astype(str)))
    #result.head()


    # ### Kmeans clustering on words
    # The clustering of keywords happen here. I have desinged this code woith capability of optimizing for number of clusters based on the minimum value of the BIC parameter.
    # We can provide minimum and maximum numbers of k and the step interval. Then the algorithm perform a grid search over the parameter space and choose the k value that minimizes BIC.
    # However, for sake of simplicity, here I just randomly choose 20 clusters and work with that.
    # The results will be saved in a data frame called: "newpd"

    # In[11]:

    print 'Running Kmeans clustering of word embeddings'
    minn=15
    maxx=16
    step=1
    results = [H2OKMeansEstimator(k=clusters, init="PlusPlus", seed=2, standardize=True) for clusters in range(minn,maxx, step)]
    for estimator in results:
        estimator.train(x=list(pd.DataFrame(result).columns), training_frame = newwd_h[1:])

    diagnostics = pd.DataFrame( [diagnostics_from_clusteringmodel(model) for model in results])
    diagnostics.set_index('Clusters', inplace=True)


    best_cluster=diagnostics[diagnostics['BIC']==diagnostics['BIC'].min()].index[0]
    print 'Number of clusters used, K: ', best_cluster
    # print results
    predicted = results[(best_cluster-minn)/step].predict(newwd_h)
    newwd_h["Cluster"] = predicted["predict"].asnumeric()
    newpd=newwd_h.as_data_frame(True)
    #newpd.head()


    # In[12]:


    """
    Here is list the word clusters, their unique word counts and their share of unique words in the whole corpus.
    """
    print 'Results: '
    member_count=newpd['Cluster'].value_counts()
    def report(a1,a2,a3):
        return 'Cluster: '+str(a1)+ ' ,Member counts: '+ str(a2)+' ,Member share: %'+ "%.2f" % a3
    print [report(member_count.index[i], member_count.iloc[i], 100.0*member_count.iloc[i]/member_count.sum()) for i in range(member_count.shape[0])]


    # ### Word Cloud
    # To show the result of our clustering I use the word cloud visualization and save the output in a pdf file with the provided filename. 
    # I also calculate the weight of each word within its own cluster and output the result in a new dataframe. 
    # Words with the highest weights in each cluster are the most frequent ones within their clusters and are shown with larger fonts.
    # Note: Although most of these words are very relevant and gives us useful intuition and information about the context and content of the corpus, 
    # it does not gaurantee every single one of them is useful! So, overall, there will be clusters or words that might not necessarly carry any usefull information abpout the content of the corpus. However, it only takes 10 seconds for a human to identify them.

    # In[38]:

    print 'Generating a pdf Wordcloud diagram "Cluster.pdf" with the top 10 words in each cluster'
    newpd2=cloud_plot(newpd, papers_trigrams,'Clusters.pdf')


    # In[39]:


    #Here I output the first top 10 words of each cluster and save it in the text file "top10words.txt" Clusters are sorted with respect to their size.
    print 'Saving the list of top 10 words for each cluster in "top10words.csv" file'
    def top10(df):
        return df.sort_values('weight',ascending=False).index.values[0:11]
    top10words=newpd2.groupby(['Cluster']).apply(top10)[member_count.index.values]
    top10words.to_csv('top10words.csv')
    print top10words


    # In[40]:

    print 'Generating a new corpus with Trigrams'
    # Since I made trigrams of the original corpus, I need to make a new corpus so I can use it in the following.
    flat_paper_id = pd.Series([item for sublist in paper_id for item in sublist])
    flat_word = pd.Series([item for sublist in papers_trigrams for item in sublist])
    newmsh=pd.DataFrame({'paper_id':flat_paper_id,'word':flat_word})
    # Here I join the result of the clustering to the new paper corpus. 
    newmsh=newmsh.set_index('word').join(newpd2,how='inner')


    # Here, I go through each paper and identify what perntage of its content is associated to each word-cluster. I output a csv file "pap_cluster_share.csv" 
    # that has paper_id, cluster id and percentage of each clsuter.
    # This is an adhoc approximation for understanding the content of each paper. For example we can say probablity of assigning paper 5633 to cluster 12, 2 and 4 is ..., ... and ...
    # Note: The best method for doing probabilistic topic modeling is to use the LDA or LDAtoVec algorithms. Here I'm doing probabilistic topic modeling with an adhoc method. 

    # In[41]:

    print 'Calculating the share of each word-cluster in each paper. Saving the results in "pap_cluster_share.csv". The top 30 rows are shown in following. '
    pap_cluster_share=newmsh.groupby(['paper_id','Cluster']).weight.sum().reset_index().rename(columns={'weight':'weight_sum'}).sort_values(['paper_id','weight_sum','Cluster'],ascending=False)
    sharepct=pap_cluster_share.groupby('paper_id').weight_sum.sum()
    pap_cluster_share=pap_cluster_share.set_index('paper_id').join(sharepct,rsuffix='_cumsum')
    pap_cluster_share['share_pct']=100.0*pap_cluster_share['weight_sum']/pap_cluster_share['weight_sum_cumsum']
    pap_cluster_share=pap_cluster_share.drop(['weight_sum','weight_sum_cumsum'],axis=1)
    pap_cluster_share.to_csv('pap_cluster_share.csv',index=True)
    print pap_cluster_share.head(30)


    # ## Another approach: Vector representation of papers. 
    # #### Trying PCA on each paper.
    # The above approach has some caveats and if we don't use it inteligently it might make our life more complicated. For example the cluster with the majority of words will always be cluster number 1 for each paper.
    # To avoid these complications and simply cluster papers into a few topics I use a different approach.
    # I laverage our FastTex embeddings and use PCA to make vector representations of papers. The I simply run a k-means clustering on the vector representation of papers.
    # I also use the BIC parameter for model selection and automatically find the optimal number of topics among papers.
    # In the following I treat each word as a feature and each FastTex columns as a data row.
    # The result is an embedding vector for each paper.

    # In[42]:

    print 'Another approach for clustering papers: Generating paper embedding vectors using the PCA analysis.'
    cols=['paper_id']+list(np.arange(1000).astype(str))
    msh2_pca=pd.DataFrame([])
    pca = PCA(n_components=1)
    for paper_id in newmsh.paper_id.unique():
        result = pca.fit_transform(newmsh[newmsh.paper_id==paper_id].loc[:,cols].T.iloc[1:,:])
        msh2_pca=pd.concat([msh2_pca,pd.DataFrame(result).rename(columns={0:str(paper_id)})],axis=1)
    msh2_pca=msh2_pca.T.reset_index().rename(columns={'index':'paper_id'})
    #msh2_pca.head()


    # In[43]:

    print 'Running Kmeans clustering on paper embedings and automatically finding the optimal number of clusters in the provided range.'
    newwd_h=h2o.H2OFrame(msh2_pca,column_names=list(msh2_pca.columns.astype(str)))
    minn=2
    maxx=20
    step=2
    results = [H2OKMeansEstimator(k=clusters, init="PlusPlus", seed=2, standardize=True) for clusters in range(minn,maxx, step)]
    for estimator in results:
        estimator.train(x=list(pd.DataFrame(msh2_pca.iloc[:,1:]).columns), training_frame = newwd_h[1:])


    diagnostics = pd.DataFrame( [diagnostics_from_clusteringmodel(model) for model in results])
    diagnostics.set_index('Clusters', inplace=True)
    diagnostics.plot(kind='line');


    best_cluster=diagnostics[diagnostics['BIC']==diagnostics['BIC'].min()].index[0]
    print 'Number of topics K ', best_cluster
    # print results
    predicted = results[(best_cluster-minn)/step].predict(newwd_h)
    newwd_h["Cluster_PCA"] = predicted["predict"].asnumeric()
    newdocs2=newwd_h.as_data_frame(True)
    #newdocs2.head()


    # #### The results of clustering, cluster ID and its member count

    # In[44]:

    print 'Results of paper clustering: '
    member_count=newdocs2['Cluster_PCA'].value_counts()
    print [report(member_count.index[i], member_count.iloc[i], 100.0*member_count.iloc[i]/member_count.sum()) for i in range(member_count.shape[0])]


    # #### Final results
    # I join the dataframe newdocs2 to our original dataframe pap to get paper titles and abstracts. I save the final clustering result as well as the paper embeddings in the Paper_Embedding_Cluster_PCA.csv file.

    # In[45]:

    print 'Joining the results to the original data file and saving the results in "Paper_Embedding_Cluster_PCA.csv" file. This file contains paper embeddings, associated clusters, titles and abstracts.'
    newdocs2=newdocs2.set_index('paper_id').join(pap[['Id','Title','Abstract']].set_index('Id'),how='inner')
    newdocs2=newdocs2.reset_index().rename(columns={'index':'paper_id'})
    newdocs2.to_csv('Paper_Embedding_Cluster_PCA.csv',index=False)


    # In[46]:

    print 'Example: Top 20 rows of saved dataframe'
    print newdocs2[['Cluster_PCA','Title']].sort_values('Cluster_PCA').head(20)


    # #### Making a TSNE 2 dimentional plot with cluster IDs

    # In[47]:

    print 'genrating a 2D TSNE visualization of paper clusters, "TSNE_papers_PCA2.pdf". Cluster IDs are used as data point labels.'
    tsne_plot(newdocs2.iloc[:,:-2].set_index('Cluster_PCA').reset_index(),'TSNE_papers_PCA2.pdf' )

    print 'Done!'
if __name__ == '__main__':
   main()
