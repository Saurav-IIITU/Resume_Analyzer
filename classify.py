from ctypes import sizeof
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import json
import pickle
import fitz

def ConvString(talklist):
  RetString = ""
  for line in talklist:
    x = line.split(' ',4)[-1]
    x = re.sub(r'\[.*\]','',x)
    x = re.sub('_1','',x)
    RetString += x + ' '
  RetString = re.sub(r'\s+',' ',RetString)
  return RetString

fname = 'AbhishekResume.pdf'
doc = fitz.open(fname)
text = ""
for page in doc:
    text = text + str(page.get_text())
tx = " ".join(text.split('\n'))
# print(tx)


filename = ['2999']
liststring = []
listtarget = []
liststring.append(tx)

filename = r'C:\Users\91823\Desktop\resume scanner new\\'
tfidf = pickle.load(open(filename+'tfidf.pickle', 'rb'))
clf = pickle.load(open(filename+'model_pkl', 'rb'))
features = tfidf.transform(liststring).toarray()
results = (clf.predict_proba(features))

res = dict(zip(clf.classes_, results[0]))
# print(str(res))
# print(clf.classes_)
# print(results)
json_object = json.dumps(res, indent = 4) 
print(json_object)