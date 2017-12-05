import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from bs4 import BeautifulSoup
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif

test = pd.read_csv('test.csv')
train= pd.read_csv('train.csv')

stops = set(stopwords.words("english"))
english_stemmer=nltk.stem.SnowballStemmer('english')
def cleanData(text,lowercase=False,remove_stops=False,stemming=False):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]',r' ',txt)
    txt = re.sub(r'\n',r' ',txt)
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in txt:
        b.append(stemmer.stem(word))

    # 5. Return a list of words
    return(b)

clean_train_reviews=[]
for txt in train['text']:
    clean_train_reviews.append("".join(cleanData(txt,True,True,True)))
    
clean_test_reviews=[]
for txt in test['text']:
    clean_test_reviews.append("".join(cleanData(txt,True,True,True)))


vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),sublinear_tf = True )
vectorizer = vectorizer.fit(clean_train_reviews)
train_features = vectorizer.transform(clean_train_reviews)
test_features = vectorizer.transform(clean_test_reviews)

fselect = SelectKBest(chi2 , k=10000)
train_features = fselect.fit_transform(train_features, train["author"])
test_features = fselect.transform(test_features)

model1 = MultinomialNB(alpha=0.001)
model1.fit( train_features, train["author"] )

pred_1 = model1.predict( test_features.toarray() )

print(pred_1)