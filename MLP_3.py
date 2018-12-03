import os, json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

with open('BOW.json') as data_file:
    word_dic = json.load(data_file)

category = ['경제', '정치', '골프', '농구', '배구', '야구', '일반스포츠', '축구', '사회', 'e스포츠']

maxlen = 100 # We will cut reviews after 100 words
max_words = word_dic.get('_MAX')
embedding_dim = 100

labels = []
sequence = []
def indexing(fdata):
    flist = []
    for txt in fdata:
        txt = txt.strip()
        word = txt.split(" ")
        for n in word:
            if n == "": continue
            if n in word_dic:
                wid = word_dic[n]
                flist.append(wid)
            else:
                print('pass', n)
    sequence.append(flist)

def label(fname):
    if fname[:2] in '경제':
        labels.append(0)
    elif fname[:2] in '정치':
        labels.append(1)
    elif fname[:2] in '골프':
        labels.append(2)
    elif fname[:2] in '농구':
        labels.append(3)
    elif fname[:2] in '배구':
        labels.append(4)
    elif fname[:2] in '야구':
        labels.append(5)
    elif fname[:5] in '일반스포츠':
        labels.append(6)
    elif fname[:2] in '축구':
        labels.append(7)
    elif fname[:2] in '사회':
        labels.append(8)
    elif fname[:4] in 'IT과학':
        labels.append(9)
    elif fname[:4] in 'e스포츠':
        labels.append(10)

for dname in category:
    dir_name = os.path.join('train_data', dname)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), 'r', encoding='utf-8')
            fdata = f.readlines()
            f.close()
            indexing(fdata)
            label(fname)

npp = np.array(sequence)
npp = pad_sequences(npp, maxlen= maxlen)
labels = to_categorical(labels)

x_train, x_val, y_train, y_val = train_test_split(npp, labels, test_size=0.9)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_data=(x_val, y_val))
model.summary()
model.save('model_MLP.h5')
