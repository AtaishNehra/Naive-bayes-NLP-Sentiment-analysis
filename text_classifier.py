#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import math
import pandas as pd
import re
import nltk
import sklearn
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
warnings.filterwarnings("ignore")
from sklearn.utils import resample
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from numpy.ma.core import log
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

filename = "values.txt"


# In[2]:


#In preprocessing we will be removing punctuation, stop words and perform lemmatisation. 
#We have used Regex(regular expressions also). We have not written the stemming part as 
#the accuracy without using stemming is more, but stemming code is mentioned as a comment in the ipynb file.
def clean_review(review):
    
    review = re.sub(r'\S*https?:\S+', '', review)
    review = re.sub(r'\S*www.\S+','',review)
    review = re.sub(r'[^\w\s]','',review)
    
    clean_1 = word_tokenize(review)

    

    
    stop_words = set(stopwords.words('english'))
    clean_2 = []
    for word in clean_1:
      if word.lower() not in stop_words:
        clean_2.append(word.lower())


    lemmatizer = WordNetLemmatizer()
    clean_4 = []
    for word in clean_2:
      clean_4.append(lemmatizer.lemmatize(word))
    
   
    return clean_4


# In[3]:


def naive_bayes_predict(review, logprior, loglikelihood):

    # First step is to get a list of words from the review we are supposed to be testing.
    word_l = clean_review(review)

    #Secondly we will initialise the probability to zero.
    total_prob = 0

    #We will now add the value of logprior to the initialised total probability.
    total_prob = logprior

    for word in word_l:

        #Here we will go through the likelihood dictionary to check if the words are present in it or not then we 
        #will add the likelihood of that word to the total probability and at the end we will print the probability using 
        #1 and 0.
        if word in loglikelihood:
            total_prob += loglikelihood[word]
            print("Token probability of",word,"is:",loglikelihood[word])

    
    if total_prob > 0:
        return 1
    else:
        return 0


# In[4]:


#We are defining a main function that will read the locally created values file (we created in ipynb code), 
#the file is mentioned above, then using the values of loglikelihood and logprior in the values file, we will
#analyse our reviews. In this main function, we are creating an application that will ask for an input, analyse it
#and give the sentiment analysis output. It will keep doing so until someone enters a capital or a small letter "x".
def main():
    with open(filename, "r") as infile:
            logprior_file = float(infile.readline())
            loglikelihood_file = ast.literal_eval(infile.readline())

    stop = 1
    while stop != "X" or stop !="x":
        review = input("Enter your Review: \n")
        classification_decision = naive_bayes_predict(review, logprior_file, loglikelihood_file)
        if classification_decision == 1:
            print("The review after analysis is negative.")
        else:
            print("The review after analysis is positive.")
        stop = input("Press 'X' to Quit or press any other key except for x, to test a new review. ")
        if stop == 'x' or stop == 'X':
            break
        else:
            continue

main()


# In[ ]:




