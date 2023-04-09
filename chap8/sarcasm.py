import tensorflow as tf
from bs4 import BeautifulSoup
import string
import urllib.request
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

#불용어 테이블
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

    # 구두점 테이블 만듬
    table = str.maketrans('', '', string.punctuation)
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

xs = []
ys = []
current_item = 1
for item in sentences:
    xs.append(current_item)
    current_item = current_item + 1
    ys.append(len(item))
newys = sorted(ys)

plt.plot(xs, newys)
plt.axis([26000, 27000, 50, 250])
plt.show()

print(newys[26000])


#훈련세트와 테스트 세트 나누기
training_size = 23000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 10000 #단어 만개로 구성된 어휘사전
max_length = 100  #최대 문장길이를 100개 단어로
trunc_type = 'post'  #이보다 문장이 길다면 끝부분을 자르고
padding_type = 'post'  #이보다 문장이 짧다면 끝에 패딩을 추가
oov_tok = "<OOV>"  #oov 토큰 사용

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
print(word_index)
training_sequences = tokenizer.texts_to_sequences(training_sentences)
print(training_sequences[0])
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(training_padded[0])

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

wc = tokenizer.word_counts
print(wc)

import matplotlib.pyplot as plt

wc = tokenizer.word_counts
from collections import OrderedDict

newlist = (OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True)))
print(word_index)
print(newlist)
xs = []
ys = []
curr_x = 1
for item in newlist:
    xs.append(curr_x)
    curr_x = curr_x + 1
    ys.append(newlist[item])

print(ys)
plt.plot(xs, ys)
plt.axis([300, 10000, 0, 100])
plt.show()
print(ys[1000])
print(ys[2000])

print(ys[3125])
print(ys[10000])
print(ys[12156])

# Need this block to get it to work with TensorFlow 2.x
import numpy as np

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
embedding_dim = 16 #각 단어에 대해 16차원의 배열을 초기화(어휘사전에 있는 각 단어는 16차원벡터에 할당됨)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim), #임베딩층 정의
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

num_epochs = 100
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_sentence(training_padded[0]))
print(training_sentences[2])
print(labels[2])

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)

print(reverse_word_index[2])
print(weights[2])

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download('vecs.tsv')
    files.download('meta.tsv')

sentences = ["granny starting to fear spiders in the garden might be real",
             "game of thrones season finale showing this sunday night", "TensorFlow book will be a best seller"]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(padded)
print(model.predict(padded))

