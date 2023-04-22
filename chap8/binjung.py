import tensorflow as tf
from bs4 import BeautifulSoup
import string
import urllib.request
import json
from tensorflow.keras.preprocessing.text import Tokenizer

#불용어 테이블
stopwords = [
    'a', 'about', 'above', 'after', 'again', 'against', 'ain\'t', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'can\'t',
    'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he',
    'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s',
    'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'just',
    'll', 'm', 'ma', 'me', 'might', 'mightn\'t', 'more', 'most', 'must', 'mustn\'t', 'my', 'myself', 'need', 'needn\'t',
    'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'oughtn\'t', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should',
    'shouldn\'t', 'so', 'some', 'such', 't', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re',
    'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who',
    'who\'s', 'whom', 'why', 'why\'s', 'will', 'with', 'won\'t', 'would', 'wouldn\'t', 'y', 'you', 'you\'d', 'you\'ll',
    'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
]


#빈정거림 기사 데이터 다운로드
url = "https://storage.googleapis.com/learning-datasets/sarcasm.json"
file_name = "sarcasm.json"
urllib.request.urlretrieve(url, file_name)

#json 화일 로딩
with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)

print(json.dumps(datastore[:100], indent=4))
print(len(datastore))

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
    #print(table)

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


#훈련세트와 테스트 세트 나누기
training_size = 23000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


max_length = 100  #최대 문장길이를 100개 단어로
trunc_type = 'post'  #이보다 문장이 길다면 끝부분을 자르고
padding_type = 'post'  #이보다 문장이 짧다면 끝에 패딩을 추가
oov_tok = "<OOV>"  #oov 토큰 사용


print(len(sentences))
print(sentences[0])
print(len(training_sentences))
tokenizer = Tokenizer(oov_token=oov_tok)
#tokenizer.fit_on_texts(training_sentences)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))
exit()
# 토크나이저 저장
import pickle
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


training_sequences = tokenizer.texts_to_sequences(training_sentences)
# print(training_sequences[0])
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# print(training_padded[0])

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
embedding_dim = 16 #각 단어에 대해 16차원의 배열을 초기화(어휘사전에 있는 각 단어는 16차원벡터에 할당됨)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index)+1, embedding_dim), #임베딩층 정의
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

# 훈련
num_epochs = 100
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# 모델 저장
model.save('binjung_model')




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