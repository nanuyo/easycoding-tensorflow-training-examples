import os
import urllib.request
import zipfile
import numpy as np

# Set the URL for the GloVe Twitter embeddings zip file
url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"

# Set the path where you want to save the downloaded zip file
save_path = "glove.twitter.27B.zip"

# Check if the file already exists
if not os.path.exists(save_path):
    print("Start Downloading!, It may take a little while, so please be patient.")
    # Download the zip file and save it to the specified path
    urllib.request.urlretrieve(url, save_path)
    print("GloVe Twitter embeddings downloaded successfully!")
else:
    print("GloVe Twitter embeddings file already exists. Skipping download.")


glove_path = 'glove\glove.twitter.27B.25d.txt'

if not os.path.exists(glove_path):
    # Extract the contents of the zip file
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall("glove")
        print("GloVe Twitter embeddings file extracted successfully!")

# Get the list of files in the extracted folder
file_list = os.listdir("glove")

# Print the list of files
print("List of files in the glove folder:")
for file_name in file_list:
    print(file_name)


#데이터를 파이썬 딕셔너리로 만듬
glove_embeddings = dict()
f = open('glove/glove.twitter.27B.25d.txt', 'r', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs

f.close()

# print(glove_embeddings['airplane'])
print(len(glove_embeddings))

##################################################################

import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import string
import tensorflow as tf

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

table = str.maketrans('', '', string.punctuation)


#빈정거림 기사 데이터 다운로드
url = "https://storage.googleapis.com/learning-datasets/sarcasm.json"
file_name = "sarcasm.json"
urllib.request.urlretrieve(url, file_name)

#json 화일 로딩
with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)


sentences = []
labels = []
urls = []
for item in datastore:
    sentence = item['headline'].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

vocab_size = 13200
embedding_dim = 25
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 23000

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector