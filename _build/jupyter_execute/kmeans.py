#!/usr/bin/env python
# coding: utf-8

# # UAS PPW

# ## Crawling data
# Crawlling merupakan proses pengambilan data yang tersedia secara online untuk umum. Proses pengambilan data informasi pada halaman menggunakan URL (Uniform Resource Locator). URL ini akan menjadi acuan untuk mencari semua hyperlink yang ada pada website. Kemudian dilakukan indexing untuk mencari kata dalam dokumen pada setiap link yang ada. Lalu jika, menyertakan API (Application Programming Interface) dapat melakukan penambangan dataset yang lebih besar. Selain itu, dengan API kita dapat mengumpulkan data lebih spesifik sesuai dengan link URL yang ada tanpa harus mengetahui element HTML pada sebuah website.

# ### Instalasi Scrapy
# Untuk menginstall scrapy dapat menggnakan perintah pip di Command Prompt berikut:

# In[1]:


pip install scrapy


# ### Membuat File Scrapy
# Setelah selesai menginstall Scrapy, maka buat file scrapy baru dengan kode:

# In[2]:


scrapy startproject namaproject


# ### Menjalankan Spyder baru
# Yang harus dilakukan, masuk ke dalam projectscrapy kemudian membuat spider baru dengan kode:

# In[ ]:


cd nama project


# Setelah menjalankan spyder baru, tuliskan command yang isinya nama file dan alamat website yang akan diambil datanya agar lebih mudah saat akan membuat.

# In[ ]:


scrapy genspider example example.com


# ### Menulis Program Scapper
# Setelah itu, pada pembuatan spyder file yang tadi telah dibuat telah tersedia dengan nama file yang telah dibuat.
# 
# Pada file yang telah dibuat akan ada code untuk melakukan scraping yang dapat diubah sesuai keinginan

# In[ ]:


import scrapy
class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://pta.trunojoyo.ac.id/welcome/detail/080211100070',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211200001',
            'https://pta.trunojoyo.ac.id/welcome/detail/080211100050',
            'https://pta.trunojoyo.ac.id/welcome/detail/100211200002',
            'https://pta.trunojoyo.ac.id/welcome/detail/080211100044',
            'https://pta.trunojoyo.ac.id/welcome/detail/080211100119',
            'https://pta.trunojoyo.ac.id/welcome/detail/080211100103',
            'https://pta.trunojoyo.ac.id/welcome/detail/080211100098',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211100079',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211100089',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211100013',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211100020',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211100064',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211100064',
            'https://pta.trunojoyo.ac.id/welcome/detail/090211100018'
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


# ### Menjalankan File Spider
# Pertama, masuk kedalam direktori spider dulu. setelah itu, jalankan spider dengan command

# In[ ]:


scrapy runspider namafile.py


# ### Menyimpan data kedalam csv
# Data yang telah diambil dari suatu web dapat disimpan dengan kode:

# In[ ]:


scrapy crawl namafile -O namafileyangdiinginkan.xlsx


# ## Preprocessing data
# Preprocessing data adalah tahapan perbersihan data dari suatu kalimat atau kata, link, atau hal –hal yang tidak diperlukan untuk melakukan analisis sentiment. Dataset yang diperoleh dari proses crawling masih berbentuk kata atau kalimat yang tidak beraturan atau tidak terstruktur, sehingga membutuhkan proses yang disebut preprocessing data agar menghasilkan data yang bersih supaya mempermudah proses analisis. Lalu pada Preprocessing sendiri melibatkan validasi dan imputasi data. Tujuan dari validasi adalah untuk menilai tingkat kelengkapan dan akurasi data yang tersaring. Preprocessing data sangat penting karena kesalahan, redundan, missing value, dan data yang tidak konsisten menyebabkan berkurangnya akurasi hasil analisis. Jadi, sebelum mengolah data, kita harus memastikan bahwa data yang akan kita gunakan merupakan data "bersih". Ada beberapa cara yang bisa digunakan untuk membersihkan data, tergantung dari jenis masalah yang ada dalam kumpulan data. Adapun tahapan preprocessing data terdiri dari Cleaning data, Case Folding, Tokenizing, Stopword, dan Stemming.

# ### Installasi library
# lakukan installasi library yang diperlukan, jika telah terinstall maka langsung panggil saja 

# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install numpy')
get_ipython().system('pip install nltk')


# In[3]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 

# ### Import library yang digunakan
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


# ### Data set
# berikut data yang digunakan pada program saat ini

# In[2]:


df=pd.read_csv('scrapylsa.csv')


# In[3]:


df.head(10)


# ### Drop data yang tidak digunakan

# In[4]:


df.drop(['judul','penulis', 'dosen_pembimbing_1', 'dosen_pembimbing_2'],axis=1, inplace=True)


# In[5]:


df.head(15)


# ### Remove angka
# pada proses ini penghapusan angka dan tanda-tanda baca yang ada pada data

# In[6]:


import string 
import re

def remove(text):
    return re.sub(r"\d+","",text)


# In[7]:


df['abstrak_remove']=df['abstrak'].apply(remove)
df.head()


# ### Cleansing data
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
df['abstrak_cleaned']=df['abstrak_remove'].apply(clean_text)


# In[10]:


df.head()


# ### Menghapus data yang tidak dibutuhkan
# hal ini bertujuan agar menggunakan data yang telah di cleansing data

# In[11]:


df.drop(['abstrak', 'abstrak_remove'],axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


df['abstrak_cleaned'][0]


# ### MENGEKSTRAK FITUR DAN MEMBUAT DOCUMENT-TERM-MATRIX ( DTM )
# Dalam DTM nilainya adalah nilai TFidf. Term Frequency — Inverse Document Frequency atau TFIDF adalah suatu metode algoritma yang berguna untuk menghitung bobot setiap kata yang umum digunakan. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat. Secara sederhana, metode TF-IDF digunakan untuk mengetahui berapa sering suatu kata muncul di dalam dokumen. Contoh yang dibahas kali ini adalah mengenai penentuan urutan peringkat data berdasarkan query yang digunakan.
# 
# Inti utama dari algoritma ini adalah melakukan perhitungan nilai TF dan nilai IDF dari setiap kata kunci terhadap masing-masing dokumen dalam korpus. 
# 
# Term Frequency (TF) yaitu pembobotan/weight setiap kata (term) pada suatu dokumen berdasarkan jumlah kemunculannya dalam dokumen tersebut. Semakin besar jumlah kemunculan suatu kata dalam dokumen, maka semakin besar pula bobot yang diberikan (TF Tinggi) jadi nilai tertinggi merupakan  jumlah kemunculan/frekuensi.
# 
# Setelah menentukan Tf maka selanjutnya kita tentukan nilai IDF nya dapat dihitung dengan rumus :
# $$
# \operatorname{idf}=\log \left(\frac{D}{d f}\right)
# $$
# 
# Selanjutnya adalah melakukan perkalian antara nilai TF dan IDF untuk mendapatkan jawaban akhir. untuk rumusnya sebagai berikut:
# $$
# \begin{gathered}
# Tf-Idf=t f_{i j} * i d f_{j} \\
# Tf-Idf=t f_{i j} * \log \left(\frac{D}{d f}\right)
# \end{gathered}
# $$
# 
# Keterangan :
# Dimana D adalah jumlah semua dokumen dalam koleksi sedangkan df adalah jumlah dokumen yang mengandung term tertentu.
# 
# Parameter dari vectorizer Tfidf mmiliki beberapa poin penting:
# 1) LSA umumnya diimplementasikan dengan nilai Tfidf di mana-mana dan tidak dengan Count Vectorizer.
# 
# 2) max_features tergantung pada daya komputasi Anda dan juga pada eval. metrik (skor koherensi adalah metrik untuk model topik). Coba nilai yang memberikan evaluasi terbaik. metrik dan tidak membatasi kekuatan pemrosesan.
# 
# 3) Nilai default untuk min_df & max_df bekerja dengan baik.
# 
# 4) Dapat mencoba nilai yang berbeda untuk ngram_range.

# In[14]:


from sklearn.feature_extraction.text import CountVectorizer

document = df['abstrak_cleaned']
a=len(document)

# Create a Vectorizer Object
vectorizer = CountVectorizer()

vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray())


# In[15]:


a = vectorizer.get_feature_names()


# ### TF-IDF

# In[16]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tf = tfidf.fit_transform(vectorizer.fit_transform(document)).toarray()


# In[27]:


dfb = pd.DataFrame(data=tf,index=list(range(1, len(tf[:,1])+1, )),columns=[a])
dfb


# ## K-Means

# In[18]:


from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[28]:


#--- Mengubah Variabel Data Frame Menjadi Array ---
x_array =  np.array(dfb)
print(x_array)


# ## LSA
# Latent Semantic Analysis (LSA) merupakan sebuah metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi. LSA pada dasarnya adalah dekomposisi nilai tunggal.
# 
# Singular Value Decomposition (SVD) adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan term-document matrix. SVD merupakan teorema aljabar linier yang menyebutkan bahwa persegi panjang dari term-document matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu :
# 
# – Matriks ortogonal U (matriks dokumen-topik)
# 
# – Matriks diagonal S (Matrik diagonal dengan elemen matriks positif atau nol)
# 
# – Transpose dari matriks ortogonal V (matriks topik-term)
# 
# Yang dirumuskan dengan :
# $$
# A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T}
# $$
# 
# Keterangan : 
# A = Matriks Masukan (Pada Penelitian matriks ini berisi matrik hasil perhitungan TF-IDF)
# 
# U = Matriks Ortogonal U
# 
# S = Matriks Diagonal S (matriks positif atau nol)
# 
# V =  Transpose Ortogonal V
# 
# 
# Setiap baris dari matriks U (matriks istilah dokumen) adalah representasi vektor dari dokumen yang sesuai. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk suku-suku dalam data kami dapat ditemukan dalam matriks V (matriks istilah-topik).
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. Kami kemudian dapat menggunakan vektor-vektor ini untuk menemukan kata-kata dan dokumen serupa menggunakan metode kesamaan kosinus.
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kita ekstrak. Model tersebut kemudian di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.
# 
# Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang terakhir kadang-kadang digunakan dalam konteks pencarian informasi.

# In[58]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# Mirip dengan dokumen lain kita bisa melakukan ini. Namun perhatikan bahwa nilai tidak menambah 1 seperti di LSA itu bukan kemungkinan topik dalam dokumen.

# In[59]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[60]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[61]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sekarang bisa mendapatkan daftar kata-kata penting dari masing-masing untuk 10 topik seperti yang ditunjukkan. Untuk kesederhanaan di sini saya telah menunjukkan 10 kata untuk setiap topik

# In[62]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# 

# In[ ]:




