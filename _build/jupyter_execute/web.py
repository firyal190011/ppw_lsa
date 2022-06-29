#!/usr/bin/env python
# coding: utf-8

# # Latent Semantic Analysis (LSA)

# Latent Semantic Analysis (LSA) merupakan sebuah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi

# # Crawling data
# Crawlling merupakan proses pengambilan data yang tersedia secara online untuk umum. Proses pengambilan data informasi pada halaman menggunakan URL (Uniform Resource Locator). URL ini akan menjadi acuan untuk mencari semua hyperlink yang ada pada website. Kemudian dilakukan indexing untuk mencari kata dalam dokumen pada setiap link yang ada. Lalu jika, menyertakan API (Application Programming Interface) dapat melakukan penambangan dataset yang lebih besar. Selain itu, dengan API kita dapat mengumpulkan data lebih spesifik sesuai dengan link URL yang ada tanpa harus mengetahui element HTML pada sebuah website.

# ## Instalasi Scrapy
# Untuk menginstall scrapy dapat menggnakan perintah pip di Command Prompt berikut:

# In[1]:


pip install scrapy


# ## Membuat File Scrapy
# Setelah selesai menginstall Scrapy, maka buat file scrapy baru dengan kode:

# In[ ]:


scrapy startproject namaproject


# ## Menjalankan Spyder baru
# Yang harus dilakukan, masuk ke dalam projectscrapy kemudian membuat spider baru dengan kode:

# In[ ]:


cd nama project


# Setelah menjalankan spyder baru, tuliskan command yang isinya nama file dan alamat website yang akan diambil datanya agar lebih mudah saat akan membuat.

# In[ ]:


scrapy genspider example example.com


# ## Menulis Program Scapper
# Setelah itu, pada pembuatan spyder file yang tadi telah dibuat telah tersedia dengan nama file yang telah dibuat.
# 
# Pada file yang telah dibuat akan ada code untuk melakukan scraping yang dapat diubah sesuai keinginan

# In[ ]:


import scrapy
class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://pta.trunojoyo.ac.id/welcome/detail/160411100077',
            'https://pta.trunojoyo.ac.id/welcome/detail/160211100071',
            'https://pta.trunojoyo.ac.id/welcome/detail/160531100046',
            'https://pta.trunojoyo.ac.id/welcome/detail/160521100004',
            'https://pta.trunojoyo.ac.id/welcome/detail/160541100008',
            'https://pta.trunojoyo.ac.id/welcome/detail/150721100036',
            'https://pta.trunojoyo.ac.id/welcome/detail/160721100026',
            'https://pta.trunojoyo.ac.id/welcome/detail/160421100099',
            'https://pta.trunojoyo.ac.id/welcome/detail/150721100026',
            'https://pta.trunojoyo.ac.id/welcome/detail/160521100008',
            'https://pta.trunojoyo.ac.id/welcome/detail/160421100169',
            'https://pta.trunojoyo.ac.id/welcome/detail/160211100041',
            'https://pta.trunojoyo.ac.id/welcome/detail/160311100017',
            'https://pta.trunojoyo.ac.id/welcome/detail/130611100192',
            'https://pta.trunojoyo.ac.id/welcome/detail/160521100043'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # print(response.url)
        yield {
            'judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'penulis': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(2) > span::text').extract(),
            'dosen_pembimbing_1': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(3) > span::text').extract(),
            'dosen_pembimbing_2': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(4) > span::text').extract(),
            'abstrak': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
        }
        # content_journal > ul > li:nth-child(1) > div:nth-child(1) > a
        # content_journal > ul > li:nth-child(1) > div:nth-child(1) > a


# ## Menjalankan File Spider
# Pertama, masuk kedalam direktori spider dulu. setelah itu, jalankan spider dengan command

# In[ ]:


scrapy runspider namafile.py


# ## Menyimpan data kedalam csv
# Data yang telah diambil dari suatu web dapat disimpan dengan kode:

# In[ ]:


scrapy crawl namafile -O namafileyangdiinginkan.xlsx


# # Preprocessing data
# Preprocessing data adalah tahapan perbersihan data dari suatu kalimat atau kata, link, atau hal –hal yang tidak diperlukan untuk melakukan analisis sentiment. Dataset yang diperoleh dari proses crawling masih berbentuk kata atau kalimat yang tidak beraturan atau tidak terstruktur, sehingga membutuhkan proses yang disebut preprocessing data agar menghasilkan data yang bersih supaya mempermudah proses analisis. Lalu pada Preprocessing sendiri melibatkan validasi dan imputasi data. Tujuan dari validasi adalah untuk menilai tingkat kelengkapan dan akurasi data yang tersaring. Preprocessing data sangat penting karena kesalahan, redundan, missing value, dan data yang tidak konsisten menyebabkan berkurangnya akurasi hasil analisis. Jadi, sebelum mengolah data, kita harus memastikan bahwa data yang akan kita gunakan merupakan data "bersih". Ada beberapa cara yang bisa digunakan untuk membersihkan data, tergantung dari jenis masalah yang ada dalam kumpulan data. Adapun tahapan preprocessing data terdiri dari Cleaning data, Case Folding, Tokenizing, Stopword, dan Stemming.

# ## Installasi library
# lakukan installasi library yang diperlukan, jika telah terinstall maka langsung panggil saja 

# In[ ]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install numpy')
get_ipython().system('pip install nltk')


# In[ ]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 

# ## Import library yang digunakan
# memanggil library-library yang digunakan

# In[1]:


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


# ## Data set
# berikut data yang digunakan pada program saat ini

# In[2]:


df=pd.read_csv('jurnal.csv')


# In[3]:


df.head(10)


# ## Drop data yang tidak digunakan

# In[4]:


df.drop(['Judul','Penulis', 'Dosbing_1', 'Dosbing_2'],axis=1, inplace=True)


# In[5]:


df.head(30)


# ## Remove angka
# pada proses ini penghapusan angka dan tanda-tanda baca yang ada pada data

# In[6]:


import string 
import re

def remove(text):
    return re.sub(r"\d+","",text)


# In[7]:


df['Abstrak_indo_remove']=df['Abstrak_indo'].apply(remove)
df.head()


# ## Cleansing data
# 
#   Data cleansing atau yang disebut juga dengan data scrubbing merupakan suatu proses analisa mengenai kualitas dari data dengan mengubah. Bisa juga pengelola mengoreksi ataupun menghapus data tersebut. Data yang dibersihkan tersebut adalah data yang salah, rusak, tidak akurat, tidak lengkap dan salah format. Selain itu, Cleaning data adalah proses menghilangkan atau  menghapus  data  dari  suatu  kalimat  yang  memiliki unsur, seperti username (@username), hastag (#), URL, angka, kalimat yang redundance,emoticon (:@, :D ), dan tanda baca  yang tidak diperlukan pada saat proses analisis sentiment. Pada proses cleaningini pun juga terjadi proses perubahan data ke lowercase.
# 

# In[8]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[9]:


# time taking
df['Abstrak_indo_cleaned']=df['Abstrak_indo_remove'].apply(clean_text)


# In[10]:


df.head()


# ## Menghapus data yang tidak dibutuhkan
# hal ini bertujuan agar menggunakan data yang telah di cleansing data

# In[11]:


df.drop(['Abstrak_indo', 'Abstrak_indo_remove'],axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


df['Abstrak_indo_cleaned'][0]


# ## TF-IDF
# TF — IDF adalah suatu metode algoritma yang berguna untuk menghitung bobot setiap kata yang umum digunakan. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat. Metode ini akan menghitung nilai Term Frequency (TF) dan Inverse Document Frequency (IDF) pada setiap token (kata) di setiap dokumen dalam korpus. Secara sederhana, metode TF-IDF digunakan untuk mengetahui berapa sering suatu kata muncul di dalam dokumen.

# Term Frequency (TF) yaitu pembobotan/weight setiap kata (term) pada suatu dokumen berdasarkan jumlah kemunculannya dalam dokumen tersebut. Semakin besar jumlah kemunculan suatu kata dalam dokumen, maka semakin besar pula bobot yang diberikan (TF Tinggi) jadi nilai tertinggi merupakan  jumlah kemunculan/frekuensi.
# 
# Setelah menentukan Tf maka selanjutnya kita tentukan nilai IDF nya dapat dihitung dengan rumus :
# $$
# \begin{gathered}
# Tf-Idf=t f_{i j} * i d f_{j} \\
# Tf-Idf=t f_{i j} * \log \left(\frac{D}{d f}\right)
# \end{gathered}
# $$
# 
# Catatan
# 
# Df = jumlah dokumen yang didalamnya memuat term tertentu
# 
# D = Jumlah Dokumen yang di perbandingkan

# In[14]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) 
# to play with. min_df,max_df,max_features etc...


# In[15]:


vect_text=vect.fit_transform(df['Abstrak_indo_cleaned'])


# In[16]:


print(vect_text.shape)
print(vect_text)


# In[17]:


vect.get_feature_names()


# In[18]:


vect_text.todense()


# In[19]:


df = pd.DataFrame(vect_text.todense().T, index=vect.get_feature_names(), columns=[f'{i+1}' for i in range (len(df))])
df


# # LSA

# In[20]:


idf=vect.idf_


# In[21]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['yang'])
print(dd['wajah'])  # police is most common and forecast is least common among the news headlines.


# In[22]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# Mirip dengan dokumen lain kita bisa melakukan ini. Namun perhatikan bahwa nilai tidak menambah 1 seperti di LSA itu bukan kemungkinan topik dalam dokumen.

# In[23]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[24]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[25]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sekarang bisa mendapatkan daftar kata-kata penting dari masing-masing untuk 10 topik seperti yang ditunjukkan. Untuk kesederhanaan di sini saya telah menunjukkan 10 kata untuk setiap topik

# In[26]:


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




