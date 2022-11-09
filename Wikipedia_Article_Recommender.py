#!/usr/bin/env python
# coding: utf-8

# ### Wikipedia Article Recommendation
# 
# #### Task :-
# Suggest some random Wikipedia articles based on user preferences 
# 
# 1. Take input from users their preferences, for example (movies, general knowledge, etc)
# 2. Suggest an article if the user wants to read it show the article on the browser else repeat the process.

# #### Dataset Used :-
# 
# For doing the task, we need a dataset that is having wikipedia's articles along with it's titles.Thus, we are going to download some random articles  from wikipedia using Python API.
# We have used only 10000 artices as of now as it's little bit time consuming but more articles we will use, better wil be our model.
# 
# 
# #### Metrics Used :-
# 
# We are using **cosine similarity** metric for finding similarity between the features from wikipedia article.
# 
# #### Why cosine similarity used?
# 
# If two vectors are far apart they might be similar as their orientation is same .In cosine similarity, even if distances are large between two vectors but if their orientation is same, it will be high.

# In[1]:


import wikipedia
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore') 


# #### Data Download

# In[2]:


data_folder = './data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
random_articles_downloaded = 500
articles = []
for each_index in range(int(random_articles_downloaded // 500)): 
    articles.append(wikipedia.random(500))
articles_titles = [article for i in articles for article in i]
print("number of random article titles:", len(set(articles_titles)))       


# #### Retrieving summary text according to the titles.

# In[3]:


titles_summary_map = dict()
for title_num in range(len(articles_titles)):  
    title = articles_titles[title_num]
    try:
        summary_text = wikipedia.summary(title)
    except:
        continue
    ## Data Cleaning. Replacing newline character,and punctuation with empty space.
    preprocessed_text = summary_text.replace("\n","").replace(";","").replace("=","").replace("/","").replace("?"," ")
    titles_summary_map[title] = preprocessed_text  


# In[4]:


all_titles_contents = list(titles_summary_map.values())
all_titles = list(titles_summary_map.keys())


# #### Applying tfidf vectorizer for converting text data into vectors. We are taking maximum features as 1500. We can take also increase this number.

# In[5]:


## Initiate TfidfVectorizer
vectorizer = TfidfVectorizer(input = all_titles_contents, lowercase = True, 
                              stop_words = "english", ngram_range = (1,5) ,max_features = 1500)
## Creation of vectors from processed text.
titles_contents_matrix = vectorizer.fit_transform(all_titles_contents)

print("number of tfidf vectorized elements:", len(vectorizer.get_feature_names()))


# #### After conversion to vectors,creating cosine similarity matrix.

# In[6]:


similarity_mat = cosine_similarity(titles_contents_matrix)


# #### Retrieve summary data from user input title and preprocessing it.

# In[35]:


def article_prediciton(input_article_title,recommended_articles):
    print("Title of the input article is: {}".format(input_article_title))
    summary_text = wikipedia.summary(input_article_title)    
    preprocessed_text = summary_text.replace("\n","").replace("?","").replace(";","").replace("=","")
    summary_text_processed_vec = vectorizer.transform([preprocessed_text])
    cos_similarity = cosine_similarity(summary_text_processed_vec,titles_contents_matrix)
    titles_retrived = similarity_mat.argsort()[0][-recommended_articles:][::-1]
    cos_similarity.sort()
    titles_cos_sim = similarity_mat[0][-recommended_articles:][::-1]
    output_titles = [all_titles[each_value] for each_value in titles_retrived ]
    print("Titles of recommended articles:{}".format(output_titles))

    for i in range(len(output_titles)):
        rec_title_value = output_titles[i]
        wiki_link_format = "https://en.wikipedia.org/wiki"
        wiki_article_link_list = wiki_link_format.split('/').copy()
        print("Recommended article number : {} \n".format(i+1))
        print("Title: {}\n".format(output_titles[i]))
        link = "/".join(wiki_article_link_list+["_".join(output_titles[i].split(" "))])
        print("Link: {}\n".format(link))


# In[39]:


user_input = 'General Knowledge'
recommended_article_number = 5
article_prediciton(user_input,recommended_article_number)


# In[41]:


user_input = 'Movie'
recommended_article_number = 3
article_prediciton(user_input,recommended_article_number)


# In[ ]:




