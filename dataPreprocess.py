#----------test gpt neo colab data prepare------------
#I used these two part seperately
#write test files
import os
from pathlib import Path
# f=open("test.txt")
# line=f.readline()
# n=0
# while line:
#     print(line)
#     tmp = str(n)
#     if line[9] == '2':
#         p = os.getcwd()
#         p = p + '/test/pos/' + tmp + '.txt'
#         with open(p,'a') as f1:
#             f1.write(line[11:-1])
#     if line[9] == '1':
#         p = os.getcwd()
#         p = p + '/test/neg/' + tmp + '.txt'
#         with open(p,'a') as f1:
#             f1.write(line[11:-1])
#     line = f.readline()
#     # if n == 200:
#     #     break
#     n=n+1
# f.close()
#--------------------------------------------------------
#generate train files
from pathlib import Path
f = open("train.txt")
line = f.readline()
folds = 1
n = 0
while line:
    print(line)
    tmp = str(n)
    if line[9] == '2':
        p = os.getcwd()
        tmp_path = p + '/data/train/train' + str(folds) + '/pos/'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        tmp_path = tmp_path + tmp + '.txt'
        with open(tmp_path,'a') as f1:
            f1.write(line[11:-1])
    if line[9] == '1':
        p = os.getcwd()
        tmp_path = p + '/data/train/train' + str(folds) + '/neg/'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        tmp_path = tmp_path + tmp + '.txt'
        with open(tmp_path,'a') as f1:
            f1.write(line[11:-1])
    line = f.readline()
    if n > 10000:
        n = 0
        folds = folds + 1
    else:
        n = n+1
    # if folds == 2:
    #     break
f.close()
#--------------------------------------------
#--------------------------------------------------------
#generate test files
from pathlib import Path
f = open("test.txt")
line = f.readline()
folds = 1
n = 0
while line:
    print(line)
    tmp = str(n)
    if line[9] == '2':
        p = os.getcwd()
        tmp_path = p + '/data/test/test' + str(folds) + '/pos/'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        tmp_path = tmp_path + tmp + '.txt'
        with open(tmp_path,'a') as f1:
            f1.write(line[11:-1])
    if line[9] == '1':
        p = os.getcwd()
        tmp_path = p + '/data/test/test' + str(folds) + '/neg/'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        tmp_path = tmp_path + tmp + '.txt'
        with open(tmp_path,'a') as f1:
            f1.write(line[11:-1])
    line = f.readline()
    if n > 5000:
        n = 0
        folds = folds + 1
    else:
        n = n+1
    # if folds == 2:
    #     break
f.close()