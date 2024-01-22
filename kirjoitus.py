import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('./data/ESKO.csv', encoding = 'iso-8859-1', sep=';')

b = ''
with open('./kaannos2.txt', 'r') as f:
    b = f.read()
    
c = b.split('\n')[:-1]

df['translate'] = c

df['main'] = df['dim_hoidonsyy'].str[0]

luokat = df[df['main'] != '-']
luokat = luokat[['translate', 'main']]

base_dir = './Data/'
train_dir = base_dir + 'train/'
test_dir = base_dir + 'test/'

for letter in luokat['main']:
    train_folder = train_dir + letter
    test_folder = test_dir + letter
    if os.path.exists(train_folder):
        continue
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    kirjain = luokat[luokat['main'] == letter]
    jako80 = int(kirjain.shape[0] * 0.8)
    indeksit = list(kirjain.index)
    for x in range(jako80):
        file = f'{train_folder}/{indeksit[x]}.txt'
        f = open(file, "w")
        f.write(kirjain.iloc[x, 0])
        f.close()
        
    for x in range(jako80, kirjain.shape[0]):
        file = f'{test_folder}/{indeksit[x]}.txt'
        f = open(file, "w")
        f.write(kirjain.iloc[x, 0])
        f.close()
