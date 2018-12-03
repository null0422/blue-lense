import os, csv
from konlpy.tag import Twitter

twitter = Twitter()
os.chdir("test_data")



category = ['IT과학', '경제', '정치', 'e스포츠', '골프', '농구', '배구', '야구', '일반스포츠', '축구', '사회']

for category_element in category:
    i=1
    file = open('Article_'+category_element+'.csv', 'r',encoding='euc-kr', newline="")
    line = csv.reader(file)
    try:
        for line_text in line:
            temp = []
            f = open(os.path.join(category_element, category_element) + '_' + str(i) + '.txt', 'w')
            for text in line_text:
                sentence = twitter.pos(text, norm=True, stem=True)
                for morpheme in sentence:
                    temp.append(morpheme)
            for j in temp:
                if "".join(j[1]) == 'Noun':
                    f.write("".join(j[0]) + ' ')
                if "".join(j[1]) == 'Number':
                    f.write("".join(j[0]) + ' ')
                if "".join(j[1]) == 'Alpha':
                    f.write("".join(j[0]) + ' ')
            i += 1
    except:
        pass


    print(category_element + ' is done!')

f.close()
