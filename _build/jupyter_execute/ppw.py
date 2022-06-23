#!/usr/bin/env python
# coding: utf-8

# #LSA#

# In[ ]:





# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install numpy')
get_ipython().system('pip install nltk')


# In[2]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# #Preprocessing data
# Preprocessing data adalah tahapan perbersihan data dari suatu kalimat atau kata, link, atau hal –hal yang tidak diperlukan untuk melakukan analisis sentiment. Dataset yang diperoleh dari proses crawling masih berbentuk kata atau kalimat yang tidak beraturan atau tidak terstruktur, sehingga membutuhkan proses yang disebut preprocessing data agar menghasilkan data yang bersih supaya mempermudah proses analisis. Lalu pada Preprocessing sendiri melibatkan validasi dan imputasi data. Tujuan dari validasi adalah untuk menilai tingkat kelengkapan dan akurasi data yang tersaring. Preprocessing data sangat penting karena kesalahan, redundan, missing value, dan data yang tidak konsisten menyebabkan berkurangnya akurasi hasil analisis. Jadi, sebelum mengolah data, kita harus memastikan bahwa data yang akan kita gunakan merupakan data "bersih". Ada beberapa cara yang bisa digunakan untuk membersihkan data, tergantung dari jenis masalah yang ada dalam kumpulan data. Adapun tahapan preprocessing data terdiri dari Cleaning data, Case Folding, Tokenizing, Stopword, dan Stemming.

# In[ ]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# #Data set

# In[ ]:


df=pd.read_csv('jurnal.csv', usecols =['Abstrak_indo'])


# In[ ]:


df.head(10)


# #Cleansing data
# 
#   Data cleansing atau yang disebut juga dengan data scrubbing merupakan suatu proses analisa mengenai kualitas dari data dengan mengubah. Bisa juga pengelola mengoreksi ataupun menghapus data tersebut. Data yang dibersihkan tersebut adalah data yang salah, rusak, tidak akurat, tidak lengkap dan salah format. Selain itu, Cleaning data adalah proses menghilangkan atau  menghapus  data  dari  suatu  kalimat  yang  memiliki unsur, seperti username (@username), hastag (#), URL, angka, kalimat yang redundance,emoticon (:@, :D ), dan tanda baca  yang tidak diperlukan pada saat proses analisis sentiment. Pada proses cleaningini pun juga terjadi proses perubahan data ke lowercase.
# 

# In[ ]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[ ]:


# time taking
df['Abstrak_indo_cleaned']=df['Abstrak_indo'].apply(clean_text)


# In[ ]:


df.head()


# In[ ]:


df.drop(['Abstrak_indo'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['Abstrak_indo_cleaned'][0]


# In[ ]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) 
# to play with. min_df,max_df,max_features etc...


# In[ ]:


vect_text=vect.fit_transform(df['Abstrak_indo_cleaned'])


# In[ ]:


print(vect_text.shape)
print(vect_text)


# In[ ]:


idf=vect.idf_


# In[ ]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['yang'])
print(dd['wajah'])  # police is most common and forecast is least common among the news headlines.


# In[ ]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[ ]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[ ]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:




