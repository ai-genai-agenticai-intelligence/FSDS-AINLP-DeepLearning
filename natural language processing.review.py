#Natural Language processing------------

#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset =pd.read_csv(r"D:\AI NLP -NATURAL LANGUAGE PROCESSING DATA\Restaurant_Reviews.tsv",delimiter='\t')


#Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus=[]

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer()
X =cv.fit_transform(corpus).toarray()

y = dataset.iloc[:-1].values


from sklearn.model_selection import train_test_split
X_train,X_teat,y_train,y_test =train_test_split(X,y ,test_size=0.20,random_state=0)


from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier()
classifier.fit_transform(X_train,y_train)

y_pred =classifier.predict(y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)
cm


#This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac =accuracy_score(y_test, y_pred)
print(ac)