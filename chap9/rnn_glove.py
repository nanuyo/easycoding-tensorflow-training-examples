import tensorflow as tf
from bs4 import BeautifulSoup
import string
import urllib.request
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

#빈정거림 기사 데이터 다운로드
url = "https://storage.googleapis.com/learning-datasets/sarcasm.json"
file_name = "sarcasm.json"
urllib.request.urlretrieve(url, file_name)

#json 화일 로딩
with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)
print(json.dumps(datastore[:10], indent=4))
print('총데이터수:', len(datastore))


# #불용어 테이블
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

# 구두점 테이블 만듬
table = str.maketrans('', '', string.punctuation)



sentences = []
labels = []
urls = []
maxlen = 0
for item in datastore:
    #headline 만 추출
    sentence = item['headline'].lower()
    #him/her => him / her 로 분리
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    #html tag 제거
    soup = BeautifulSoup(sentence, features="html.parser")
    sentence = soup.get_text()
    # print(sentence)
    #former versace store clerk sues over secret 'black code' for minority shoppers

    #문장을 단어테이블로 만듦
    words = sentence.split()
    #print(words)
    #['former', 'versace', 'store', 'clerk', 'sues', 'over', 'secret', "'black", "code'", 'for', 'minority', 'shoppers']

    filtered_sentence = ""
    for word in words:
        #구두점 제거
        word = word.translate(table)
        # 불용어 제거
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "

    sentences.append(filtered_sentence)    #'former versace store clerk sues secret black code minority shoppers '
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

print('총문장수:', len(sentences))

#전체 문장의 토큰화
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print('토큰화된 총단어수:', len(word_index))

# 토크나이저 저장
import pickle
# saving
with open('tokenizer_glove.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#훈련세트와 테스트 세트 나누기
training_size = 23000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


max_length = 100  #최대 문장길이를 100개 단어로
trunc_type = 'post'  #이보다 문장이 길다면 끝부분을 자르고
padding_type = 'post'  #이보다 문장이 짧다면 끝에 패딩을 추가

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

###############여기 까지는 8장 binjung.py 와 동일#############################

###############Glove 다운로드 #############################
import zipfile
import os

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
# glove_path = 'glove\glove.twitter.27B.100d.txt'

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
f = open(glove_path, 'r', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs

f.close()

# print(glove_embeddings['airplane'])
print(len(glove_embeddings))

##########양방향 LSTM 2층, dropout 적용하여 훈련 ###########################################
embedding_dim = 25 #glove.twitter.27B.25d.txt 는 25 차원 , glove\glove.twitter.27B.100d.txt 는 100차원
vocab_size = len(word_index)+1
# print(vocab_size)


embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

print(embedding_matrix[2])



model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True, dropout=0.2)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, dropout=0.2)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# adam = tf.keras.optimizers.Adam(learning_rate=0.00001,  beta_1=0.9, beta_2=0.999, amsgrad=False)
#과대 적합을 줄이기 위해서 손실율을 20% 낮춤
adam = tf.keras.optimizers.Adam(learning_rate=0.000008, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])
model.summary()


num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


# 모델 저장
model.save('rnn_glove_model')

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.savefig('rnn_glove' + string + '.png')
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")