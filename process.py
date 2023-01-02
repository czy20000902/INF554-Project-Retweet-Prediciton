import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit  # pip install verstack
from gensim.models import Word2Vec
from nltk.corpus import stopwords

train_data = pd.read_csv("train.csv")
output = open('sentences.csv', mode='w', encoding='UTF-8')
# stopwords = pd.read_csv("stopwords.txt")
# stopwords = stopwords[0].tolist()

sentences = train_data.values[:, 0]
print(sentences)
output.write('\"text\",' + '\n')
for i in sentences:
    output.write('\"' + i + '\",' + '\n')
