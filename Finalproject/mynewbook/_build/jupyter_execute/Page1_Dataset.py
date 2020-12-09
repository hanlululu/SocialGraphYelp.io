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


# # The yelp dataset 

# The original dataset that we have selected to analyse this problem is the [Yelp dataset](https://www.yelp.com/dataset). This dataset contains a large subset of Yelp's businesses, reviews, check-ins and user-related public anonymous data. It was originally created for the Kaggle _Yelp Dataset Challenge_ which encouraged academics to conduct research or analysis on
# the company's subset of data and show their discoveries to the world. 
# 
# The original __Yelp Academic Dataset__ contains information about 1.637.138 reviews collected from 192.609 businesses across 10 metropolitan areas in multiple countries of the world. Since this dataset is extremely large to work with (containing __10+ GB of data__), we have subsetted the data further into __one geographical area of interest: Las Vegas, US__, and __one business category: Hotels__. 
# 
# *Here we present the filtered dataframes for hotels and their reviews*
# 
# ---

# ## The Hotels Dataframe

# There are in total **438 hotels** identified by their unique business ids. The name, address, city, state, postal_code, latitude, longitude have been collected. The average stars of the hotels can be seen in the "stars" column, right next to the review_counts. 

# In[2]:


business_df = pd.read_csv('./data/las_vegas_all_business.csv')
review_df = pd.read_csv('./data/las_vegas_all_reviews.csv')
keywords = pd.read_csv('./data/las_vegas_business_keywords.csv')


# In[3]:


import warnings
warnings.filterwarnings('ignore')
# Filter business_id according to the category
filter_business_id = business_df[business_df.categories.str.contains(r"(Hotels, )|(, Hotels$)", regex = True)==True].business_id
# Filter businesses
business_df = business_df[business_df.business_id.isin(filter_business_id)].reset_index().drop('index', axis = 1).rename({'stars': 'stars_business'})
# Filter reviews based on business_id
review_df = review_df[review_df.business_id.isin(filter_business_id)].reset_index().drop('index', axis = 1).rename({'stars': 'stars_review'})

#columns = ['business_id', 'name', 'address', 'city', 'state', 'postal_code',
       #'latitude', 'longitude', 'stars', 'review_count', 'is_open',
       #'attributes', 'categories', 'hours']
#@interact
#def show_dataframe(column1=columns, column2 = columns, column3 = columns, column4 = columns):
    
    #return business_df[[column1]+[column2]+[column3]+[column4]]

business_df


# ## The Reviews Dataframe

# There are in total **172159 reviews** stored in the "text" column and the dates of reviews given stored in the "date" column, together with the ratings the users gave shown in the "stars" column. The usefulness, funiness, and coolness of the reviews are also been rated from a scale of 0 to 5. Moreover, the review, user and business ids have been collected. 

# In[4]:


#columns_rev = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny',
       #'cool', 'text', 'date']
#@interact
#def show_dataframe(column1=columns_rev, column2 = columns_rev, column3 = columns_rev, column4 = columns_rev):
    
    #return review_df[[column1]+[column2]+[column3]+[column4]]
review_df

