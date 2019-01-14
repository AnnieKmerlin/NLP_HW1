#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import math
import numpy as np
from matplotlib import pyplot as plt

import nltk, collections
from nltk import word_tokenize, sent_tokenize, bigrams
from nltk.util import ngrams

from nltk.probability import FreqDist
from nltk.corpus import stopwords


# In[2]:


#  NLTK Downloader to obtain the resource:
nltk.download('punkt')


# ### 1: Structuring the data

# #### Loading the raw data and constitute the data into a one-line-per-sentence format
# 

# In[2]:


filename = 'output.txt'

with open(filename, 'r') as file:
    # Read all text lines at once
    lines = file.read()
    # Replace New-line character with space character
    stripped = lines.replace('\n', ' ')
    # Make sentence tokens from the text
    sentences = sent_tokenize(stripped)


# In[4]:


sentences[:5]


# #### Break sentences into words, remove punctuations, and transform to uppercase 

# In[3]:


# Get all punctuations
punct_set = set(string.punctuation)
corpus = []

for sentence in sentences:
    tokens = word_tokenize(sentence.upper())
    no_punct_toks = [t for t in tokens if t not in punct_set]
    corpus.append(no_punct_toks)


# #### 1. How many sentences are there in the given corpus?

# In[5]:


sentences_num = len(corpus)
print('There are {} sentences in this corpus.'.format(sentences_num))


# ### 2: Counting and comparing

# In[4]:


# Combine all the tokens in the corpus
corpus_tokens = []
for sentence in corpus:
    corpus_tokens += sentence


# #### Unigram frequency count of each word

# In[26]:


#get the bigrams
count_unigram = corpus_tokens
#compute the frequency count
freq_unigram = collections.Counter(count_unigram)
#Have just printed the three common unigrams,the computer hangs otherwise
freq_unigram.most_common(3)


# #### Bigram frequency count of each word

# In[22]:


#get the bigrams
count_bigram = nltk.bigrams(corpus_tokens)
#compute the frequency count
freq_bigram = collections.Counter(count_bigram)
#Have just printed the three common bigrams,the computer hangs otherwise
freq_bigram.most_common(3)


# #### 1. How many unique types are present in this corpus?

# In[6]:


# Combine all the tokens in the corpus
corpus_tokens = []
for sentence in corpus:
    corpus_tokens += sentence


# In[6]:


distinct_words = set(corpus_tokens)
distinct_words_num = len(distinct_words)
print('There are {} unique types in this corpus.'.format(distinct_words_num))


# #### 2. How about unigram tokens?

# In[7]:


tokens_num = len(corpus_tokens)
print('There are {} unigram tokens present in this corpus.'.format(tokens_num))


# #### 3. Produce a rank-frequency plot (similar to those seen on the Wikipedia page for Zipf's Law) for this corpus.

# In[5]:


fdist = FreqDist(corpus_tokens)


# In[11]:


ranked_dist = fdist.most_common()
freq_log = []
rank_log = []

# Get the logs of frequencies and ranks
for rank, freq in enumerate(ranked_dist):
    # Compute the logs of frq and rank
    log_f = math.log10(freq[1])
    log_r = math.log10(rank + 1)
    
    # Append to the lists
    freq_log.append(log_f)
    rank_log.append(log_r)


# In[12]:


plt.plot(rank_log, freq_log)
plt.title("Zipf's Law Rank-frequency plot")
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')


# #### 4. What are the twenty most common words?

# In[13]:


most_common_20 = [word[0] for word in fdist.most_common(20)]
print('The twenty most common words are: {}'.format(most_common_20))


# #### 5. What happens to your type/token counts if you remove stopwords using nltk.corpora's stopwords list?

# In[9]:


stop_words = set(stopwords.words('english'))
cleaned_corpus_tokens = [t for t in corpus_tokens if t.lower() not in stop_words]


# In[10]:


distinct_words = set(cleaned_corpus_tokens)
distinct_words_num = len(distinct_words)
print('There are {} unique types in this corpus.'.format(distinct_words_num))


# In[15]:


distinct_words_num = len(cleaned_corpus_tokens)
print('There are {} unique types in this corpus with stopwords removed.'.format(distinct_words_num))


# #### 6. After removing stopwords, what are the 20 most common words?

# In[16]:


cleaned_fdist = FreqDist(cleaned_corpus_tokens)


# In[17]:


most_common_20 = [word[0] for word in cleaned_fdist.most_common(20)]
print('The twenty most common words are: {} with stopwords removed.'.format(most_common_20))


# #### Word association metrics

# #### Recalling Emily Bender's sage advice- "Look at your data!"- examine the 30 highest-PMI word pairs, along with their unigram and bigram frequencies. What do you notice?

# In[6]:


word_pair = [' '.join([pair[0], pair[1]]) for pair in bigrams(corpus_tokens)]
pair_fdist = FreqDist(word_pair)


# In[7]:




pmi = {}
tokens_num = len(corpus_tokens)

for pair in word_pair:
    # Get the freq of the pair
    w1w2_freq = pair_fdist.get(pair)
    
    # Only consider bigrams that occur with frequency above that threshold
    if w1w2_freq :
        pair_split = pair.split(' ')
        
        # Get the freq of each of the words pair
        w1_freq = fdist.get(pair_split[0])
        w2_freq = fdist.get(pair_split[1])
        
        # Compute the unigram probabilities in the corpus
        p_w1 = w1_freq / tokens_num
        p_w2 = w2_freq / tokens_num
        
        # Compute the bigram probability
        p_w1w2 = w1w2_freq / w1_freq
    
        # PMI(w1,w2)=P(w1,w2)/P(w1)P(w2)
        pmi[pair] = np.log2(p_w1w2 / (p_w1 * p_w2))


# In[8]:


pmi_sorted = sorted(pmi, key=pmi.get, reverse=True)
top_30 = pmi_sorted[:30]
print('The 30 highest-PMI word pairs are: {}.'.format(top_30))


# In[11]:


count_unigram_top30 = top_30
#compute the frequency count
freq_unigram = collections.Counter(count_unigram_top30)
print('The 30 highest-PMI word pairs and their frequencies: {}.'.format(freq_unigram))


# In[12]:


#get the bigrams
count_bigram = nltk.bigrams(top_30)
#compute the frequency count
freq_bigram = collections.Counter(count_bigram)
print('The 30 highest-PMI word pairs and their bigramfrequencies: {}.'.format(freq_bigram))


# #### Experiment with a few different threshold values, and report on what you observe

# In[12]:


# Experiment with a few different threshold values, and report on what you observe.
pmi_threshold = 100
pmi = {}
tokens_num = len(corpus_tokens)

for pair in word_pair:
    # Get the freq of the pair
    w1w2_freq = pair_fdist.get(pair)
    
    # Only consider bigrams that occur with frequency above that threshold
    if w1w2_freq > pmi_threshold:
        pair_split = pair.split(' ')
        
        # Get the freq of each of the words pair
        w1_freq = fdist.get(pair_split[0])
        w2_freq = fdist.get(pair_split[1])
        
        # Compute the unigram probabilities in the corpus
        p_w1 = w1_freq / tokens_num
        p_w2 = w2_freq / tokens_num
        
        # Compute the bigram probability
        p_w1w2 = w1w2_freq / w1_freq
    
        # PMI(w1,w2)=P(w1,w2)/P(w1)P(w2)
        pmi[pair] = np.log2(p_w1w2 / (p_w1 * p_w2))


# In[13]:


# effect of threshold on pmi
pmi


# #### With a threshold of 100, what are the 10 highest-PMI word pairs?

# In[14]:


pmi_sorted = sorted(pmi, key=pmi.get, reverse=True)
top_10 = pmi_sorted[:10]
print('The 10 highest-PMI word pairs are: {}.'.format(top_10))


# #### Examine the PMI for "New York". Explain in your own words why it is not higher.

# In[15]:


ny_pmi = pmi['NEW YORK']
print('The PMI for "New York" is: {}.'.format(ny_pmi))


# In[18]:


fdist.get('NEW')


# In[19]:


fdist.get('YORK')

