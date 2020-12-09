#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic libraries
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import itertools

# Network libraries
import networkx as nx
from fa2 import ForceAtlas2 as FA2
import community

# NLP libraries
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator

# Display libraries
from IPython.display import display
from IPython.core.display import display as disp, HTML
from ipywidgets import widgets, interact, interactive, fixed, interact_manual
import imageio

from plotly import __version__
import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
import chart_studio.plotly as py
import plotly.offline
sns.set()

pio.renderers.default = 'notebook'

business_df = pd.read_csv('./data/las_vegas_all_business.csv')
review_df = pd.read_csv('./data/las_vegas_all_reviews.csv')

import warnings
warnings.filterwarnings('ignore')
# Filter business_id according to the category
filter_business_id = business_df[business_df.categories.str.contains(r"(Hotels, )|(, Hotels$)", regex = True)==True].business_id
# Filter businesses
business_df = business_df[business_df.business_id.isin(filter_business_id)].reset_index().drop('index', axis = 1).rename({'stars': 'stars_business'})
# Filter reviews based on business_id
review_df = review_df[review_df.business_id.isin(filter_business_id)].reset_index().drop('index', axis = 1).rename({'stars': 'stars_review'})


# # NLP analysis of hotel reviews 

# Text here

# In[2]:


business_review_df = pd.merge(business_df, review_df, on='business_id', how='outer')
# Function to test if something is an adjective
is_adj = (lambda pos: pos[:2] == 'JJ')

# Function for tokenizing the text into a list of words
def tokenize_text(text):
    # Replace all the non-alphanumeric characters with space
    text = re.sub(r'[\W]', ' ', text)

    # Tokenize texts into lists of tokens
    words = word_tokenize(text)    
   
    stop_words = set(stopwords.words('english'))
       
    words = [word.lower() for word in words if word not in string.punctuation]    # Remove punctuation and set words to lowercase
    words = [word for word in words
                 if word not in stop_words
                 if len(word) > 1]    # Remove stopwords and letters (stopwords are lowercase, so this step should be done after word.lower())
    words = [word for (word, pos) in nltk.pos_tag(words) if is_adj(pos)]    # Keep adjectives only

    return words

def reviews(df, mode, dict_communities = None):
    
    if mode == 'community':
        # Create a dictionary for storing text of each community
        community_reviews = {}
        for comm in dict_communities.keys():
            community_text = ' '
            for business_id in dict_communities[comm]:
                business_text = ' '.join(df[df.business_id==business_id].text)
                community_text += business_text
            community_reviews[comm] = community_text
        return community_reviews
      
    if mode == 'business':
        # Within each business
        business_reviews = {}
        for business_id in df.business_id.unique():      
            # Concatenate all reviews of a specific business into one text
            business_text = ' '.join(df[df.business_id==business_id].text)
            business_reviews[business_id]  = business_text
        return business_reviews
    
# Function for computing TF (using different methods)
def tf(reviews, tf_method = 'term_frequency'):

    # Create a nested dictionary {business: {word: tf score}} or {business: {word: tf score}} for storing term-frequency
    term_frequency = {}

    for comm, review in reviews.items():
        # Create a dictionary for each either community or business to store words and counts of words
        term_frequency[comm] = {}

        # Total word amount for one business (for tf_method=='term_frequency')
        total_word_amount = len(review)
        # Tokenize the text into a list of words
        words = tokenize_text(review)
        # Count words
        for word in words:
            if word not in term_frequency[comm].keys():
                term_frequency[comm][word] = 1
            else:
                term_frequency[comm][word] += 1       

        # Compute different types of term frequency
        if tf_method == 'raw_count':
            term_frequency = term_frequency

        elif tf_method == 'term_frequency':
            term_frequency[comm] = {k : v/total_word_amount for k, v in term_frequency[comm].items()}

        elif tf_method == 'log':
            term_frequency[comm] = {k : math.log(1 + v) for k, v in term_frequency[comm].items()}

        elif tf_method == 'double_normalization':
            term_frequency[comm] = {k : (0.5 + 0.5*v/max(term_frequency[comm].values())) for k, v in term_frequency[comm].items()}        

    return term_frequency

# Function for computing IDF (using different methods)
def idf(reviews, term_frequency, idf_method='idf'):
    # Total number of documents (i.e. total number of businesses in this case, because we concatenate all the reviews of one specific business)
    N = len(reviews.keys())
    
    # Create a nested dictionary for {business: {word: idf score}} storing term-frequency
    inverse_document_frequency = {}
    
    for comm1 in term_frequency.keys():
        # Update the idf dictionary into form as {business: {word: 0}}
        inverse_document_frequency[comm1] = {k : 0 for k in term_frequency[comm1].keys()}
        
        for word in term_frequency[comm1]:
            # If a specific word occurs in another business, add 1 to the count of this word
            for comm2 in term_frequency.keys():
                if word in term_frequency[comm2].keys():
                    inverse_document_frequency[comm1][word] += 1
        
        # Compute different types of inverse document frequency based on the number of occurance of a word in all the businesses
        if idf_method == 'idf':
            inverse_document_frequency[comm1] = {k : math.log(N/v) for k, v in inverse_document_frequency[comm1].items()}
        elif idf_method == 'idf_smooth':
            inverse_document_frequency[comm1] = {k : (math.log(N/(1+v))+1) for k, v in inverse_document_frequency[comm1].items()}
    
    
    return inverse_document_frequency

# Function for computing TD-IDF score
def tf_idf(term_frequency, inverse_document_frequency):
    
    tf_idf = {}
    for comm in term_frequency:
        tf_idf[comm] = {k : v*term_frequency[comm][k] for k, v in inverse_document_frequency[comm].items()}
    
    return tf_idf

# Convert reviews of each business into one text for analysis
business_reviews = reviews(df = business_review_df, mode = 'business')
# Calculate term frequency of each business
business_term_frequency = tf(reviews = business_reviews, tf_method = 'term_frequency')
# Calculate inverse document frequency of each business
business_inverse_document_frequency  =  idf(reviews = business_reviews, term_frequency = business_term_frequency, idf_method = 'idf')
# Calculate TF-IDF score of each business
business_tf_idf = tf_idf(term_frequency = business_term_frequency,  inverse_document_frequency = business_inverse_document_frequency)


# Function for extracting top keywords with highest TF-IDF score of each business
def retrieve_top_n_keywords(n, tf_idf_score = business_tf_idf):
    
    # Create a dictionary, which will contain top n keywords for each business
    top_keywords_dict = {}
    
    # For each business, we will save its business_id and its top n keywords in a dictionary
    for business_id, term_dict in tf_idf_score.items():
        
        # Sort the terms by their TF-IDF score (descendingly), and keep the top n keywords
        top_n_keywords_list = [tf_idf_tuple[0] for tf_idf_tuple in sorted(term_dict.items(), key = (lambda x : x[1]), reverse = True)][0:n]
        top_keywords_dict[business_id] = list(top_n_keywords_list)
        
    return top_keywords_dict

# n = how many top keywords should the network check for nodes to have in common
n = 10
business_top_keywords = retrieve_top_n_keywords(n)

# Add a column containing top keywords of each business to the original dataframe
business_df['top_keywords'] = business_df['business_id'].map(business_top_keywords)


# In[3]:


## define node size as number of reviews 
# Filter reviews based on business_id
filter_review = review_df[review_df.business_id.isin(filter_business_id)]
hotels = business_df.sort_values(by=['business_id'])

## create a column with 3 keywords
keywords = []
for wordlist in list(hotels['top_keywords']):
    keywords.append(wordlist[:3])
hotels['keywords'] = keywords

fig = px.scatter_mapbox(hotels, lat="latitude", lon="longitude", hover_name="name", hover_data = ["keywords"], color = 'review_count',
                        color_continuous_scale=px.colors.cyclical.IceFire, size = "review_count", zoom=10, height=600)
fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

