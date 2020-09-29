# -*- coding:utf-8 -*-
import re
import os
'''
python3.5
存在bug，清洗前先清理如下的链接：
 
img src="//p3.pstatp.com/large/3198000ab35150e86405&quot
 
'''
# file 表示源文件名字，修改此处即可
file="hot_ponit_news.csv"
dirs="./result"
 
def cleandata():
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    f1 = open(file, 'r', encoding='utf-8')
    f2 = open(dirs+'./result.csv', 'a', encoding='utf-8')
 
    p1 = r"(?:[\u2E80-\uFFFD]|[\u0030-\u0039]|[\u201c-\u201d]|[\u002d]|[\u003a])+"
    pattern1 = re.compile(p1)
 
    for line in f1.readlines():
 
        matcher1 = re.findall(pattern1, line)
        str1=str()
        if matcher1:
            str1 = ' '.join(matcher1)
            f2.write(str1)
        f2.write('\n')
 
    f1.close()
    f2.close()
 
#if __name__ == '__main__':
#    cleandata()