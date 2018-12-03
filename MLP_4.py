from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import os, json
from numpy import argmax
from keras.models import load_model

with open('BOW.json') as data_file:
    word_dic = json.load(data_file)

model = load_model('model_MLP.h5')
os.chdir("test")
category = ['경제', '정치', '골프', '농구', '배구', '야구', '일반스포츠', '축구', '사회', 'e스포츠']

maxlen = 100 # We will cut reviews after 100 words

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

for fname in os.listdir():
    if fname[-4:] == '.txt':
        f = open(fname, 'r', encoding='utf-8')
        fdata = f.readlines()
        f.close()
        indexing(fdata)
        label(fname)

x_test = np.array(sequence)
x_test = pad_sequences(x_test, maxlen= maxlen)
labels = to_categorical(labels)
y_test = labels

xhat_idx = np.random.choice(x_test.shape[0], 11, replace=False)
xhat = x_test[xhat_idx]

yhat = model.predict_classes(xhat)

for i in range(len(xhat_idx)):
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))