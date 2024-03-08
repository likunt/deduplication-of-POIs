#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#import json
import re
import sys
import math
import string
#import joblib
import Levenshtein
from unidecode import unidecode
from nltk.tokenize import word_tokenize
import difflib

stop_words = ['the', 'and']

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Sentence transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def add_features(df):
    df = add_name_features(df)
    df = add_distance_features(df)
    df = add_address_features(df)
    return df

# longest common subsequence
def LCS(S, T):
    m, n = len(S), len(T)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = max(dp[i-1][j-1]+(S[i-1]==T[j-1]), dp[i][j-1], dp[i-1][j])
            
    return dp[-1][-1]

def normalized_lcs(S, T):
    m, n = len(S), len(T)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = max(dp[i-1][j-1]+(S[i-1]==T[j-1]), dp[i][j-1], dp[i-1][j])
            
    lcs = dp[-1][-1]
    return lcs/float(max(m, n))

def normalized_leven(s1, s2):
    maxlen = max(len(s1), len(s2));
    # normalize by length, high score wins
    return float(maxlen - Levenshtein.distance(s1, s2)) / float(maxlen);


def preprocessor(name):
    
    # unidecode
    name = unidecode(name)
    
    # lowercase
    name = name.lower()
    
    # remove punctuation
    name = "".join([char for char in name if char not in string.punctuation])
    
    # tokenization
  #  name_lst = word_tokenize(name) 
    name_lst = name.split(" ")

    # stopword filtering
    name_lst = [w for w in name_lst if w not in stop_words]
    
    return " ".join(name_lst)
    
def add_name_features(df):
    
    # name1 vs. name2
    geshs = [] 
    levens = []
    jaros = []
    lcss = []
    tfidf = []
    embed = []
    
    f = open("debug_name.txt", "a")
    
    j = 0
    
    for str1, str2 in df[[f"name1", f"name2"]].values.astype(str):
        if str1=="nan" or str2=="nan":
            geshs.append(np.nan)
            levens.append(np.nan)
            jaros.append(np.nan)
            lcss.append(np.nan)
            tfidf.append(np.nan)
            embed.append(np.nan)
            continue
        
        str1, str2 = preprocessor(str1), preprocessor(str2)
        # string measures
        geshs.append(difflib.SequenceMatcher(None, str1, str2).ratio())     
        levens.append(normalized_leven(str1, str2))
        jaros.append(Levenshtein.jaro_winkler(str1, str2))
        lcss.append(normalized_lcs(str1, str2))
        
        if j%1000==0:
            f.write("step:"+str(j)+"\n")
            
        j+=1
       
        # TF-IDF
        # Generate matrix of word vectors
        if str1==str2:
            tfidf.append(1.0)
        else:
            m = vectorizer.fit_transform([str1, str2])
            # compute and print the cosine similarity matrix
            cosine_sim = cosine_similarity(m, m)
            tfidf.append(cosine_sim[0][1])
        
        # SBert
        corpus_embeddings = embedder.encode([str1], convert_to_tensor=True)
     #   corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
        query_embedding = embedder.encode([str2], convert_to_tensor=True)
     #   query_embedding = util.normalize_embeddings(query_embedding)
        cos_score = util.cos_sim(query_embedding, corpus_embeddings)[0]
        embed.append(float(cos_score))

    df[f"name_gesh"] = geshs
    df[f"name_leven"] = levens
    df[f"name_jaro"] = jaros
    df[f"name_lcss"] = lcss
    df[f"name_tfidf"] = tfidf
    df[f"name_embed"] = embed
    
    f.close()
       
    return df


def add_distance_features(df):
     
    eucd = []
    geod = []

    for x1, y1, x2, y2 in df[[f"lat1", f"lon1", f"lat2", f"lon2"]].values.astype(float):

        # euclidean distance
        P1 = np.array((x1, y1))
        P2 = np.array((x2, y2))
        # subtracting both the vectors
        temp = P1 - P2
        eucd.append(np.sqrt(np.dot(temp.T, temp)))
        
        # geodesic distance
        geod.append(geodesic(P1, P2).km)

    df[f"dist_eucd"] = eucd
    df[f"dist_geod"] = geod

    return df

def add_address_features(df):
    
    # addr1 vs. addr2
    geshs = [] 
    levens = []
    jaros = []
    lcss = []
    tfidf = []
    embed = []
    
    f = open("debug_address.txt", "a")
    
    j = 0
    
    for str1, str2 in df[[f"addr1", f"addr2"]].values.astype(str):
        if str1=="nan" or str2=="nan":
            geshs.append(np.nan)
            levens.append(np.nan)
            jaros.append(np.nan)
            lcss.append(np.nan)
            tfidf.append(np.nan)
            embed.append(np.nan)
            continue
            
        str1, str2 = preprocessor(str1), preprocessor(str2)
        geshs.append(difflib.SequenceMatcher(None, str1, str2).ratio())     
        levens.append(normalized_leven(str1, str2))
        jaros.append(Levenshtein.jaro_winkler(str1, str2))
        lcss.append(normalized_lcs(str1, str2))
        
        if j%1000==0:
            f.write("step:"+str(j)+"\n")
            
        j+=1
        
        # TF-IDF
        # Generate matrix of word vectors
        if str1==str2:
            tfidf.append(1.0)
        else:
        
            # TF-IDF
            # Generate matrix of word vectors
            m = vectorizer.fit_transform([str1, str2])
            # compute and print the cosine similarity matrix
            cosine_sim = cosine_similarity(m, m)
            tfidf.append(cosine_sim[0][1])
        
        # SBert
        corpus_embeddings = embedder.encode([str1], convert_to_tensor=True)
     #   corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
        query_embedding = embedder.encode([str2], convert_to_tensor=True)
     #   query_embedding = util.normalize_embeddings(query_embedding)
        cos_score = util.cos_sim(query_embedding, corpus_embeddings)[0]
        embed.append(float(cos_score))

    df[f"addr_gesh"] = geshs
    df[f"addr_leven"] = levens
    df[f"addr_jaro"] = jaros
    df[f"addr_lcss"] = lcss
    df[f"addr_tfidf"] = tfidf
    df[f"addr_embed"] = embed    
    
    f.close()
       
    return df