import os, json

category = ['경제', '정치', '골프', '농구', '배구', '야구', '일반스포츠', '축구', '사회', 'e스포츠']

word_dic = { "_MAX": 0 }
def indexing(fdata):
    for txt in fdata:
        txt = txt.strip()
        words = txt.split(" ")
    for n in words:
        if n == "": continue
        if not n in word_dic:
            word_dic[n] = word_dic["_MAX"]
            word_dic["_MAX"] += 1
        else:
            word_dic[n]

for dname in category:
    dir_name = os.path.join('train_data', dname)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), 'r', encoding='utf-8')
            fdata = f.readlines()
            f.close()
            indexing(fdata)


with open('BOW.json', 'w') as f:
  json.dump(word_dic, f, indent=2, ensure_ascii=False)