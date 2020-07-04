import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Welcome to the NLP Project for this section of the course. In this NLP project you will be attempting to classify 
Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews. This will be a simpler 
procedure than the lecture, since we will utilize the pipeline methods for more complex tasks. 

We will use the Yelp Review Data Set from Kaggle.

Each observation in this dataset is a review of a particular business by a particular user.

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is 
better.) In other words, it is the rating of the business by the person who wrote the review. 

The "cool" column is the number of "cool" votes this review received from other Yelp users.

All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other 
words, it is a rating of the review itself, not a rating of the business. 

The "useful" and "funny" columns are similar to the "cool" column.

Let's get started! Just follow the directions below!
"""

yelp = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\20-Natural-Language-Processing"
                   r"\yelp.csv")

print('Yelp Data Frame')
print(yelp.head())
print('\n Info')
print(yelp.info())
print('\n Describe')
print(yelp.describe())
print('\n')

print('Messages')
print(yelp['text'])

yelp['text length'] = yelp['text'].apply(len)

print(yelp.describe())
print('\n')
print(yelp[yelp['text length'] == yelp['text length'].max()]['text'].iloc[0])
print('\n Nº of stars of the message')
print(yelp[yelp['text length'] == yelp['text length'].max()]['stars'])
print('\n')

sns.set_style('whitegrid')

g = sns.FacetGrid(yelp, col='stars', height=4, aspect=0.7)
g.map(plt.hist, 'text length', bins=50)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='stars', data=yelp, palette='rainbow')
plt.show()

by_star = yelp.groupby('stars').mean()
print('By Star')
print(by_star)
print('\n')

print(by_star.corr())

plt.figure(figsize=(12, 12))
sns.heatmap(data=by_star.corr(), annot=True, cmap='coolwarm')
plt.show()

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
print('\n YELP CLASS')
print(yelp_class)

X = yelp_class['text']

y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB

rate_model_detection = MultinomialNB()

rate_model_detection.fit(X_train, y_train)

predictions = rate_model_detection.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('naive', MultinomialNB())
])

X = yelp_class['text']

y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print('Without text process, but with TFITF')
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


import string
from nltk.corpus import stopwords


def text_process(mess):
    nonpunc = [char for char in mess if char not in string.punctuation]
    nonpunc = ''.join(nonpunc)
    clean = [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
    return clean


X = yelp_class['text']

y = yelp_class['stars']

mess0 = X[0]

bow0 = text_process(mess0)
print(bow0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

pipe = Pipeline([
    ('cv', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('naive', MultinomialNB())
])


pipe.fit(X_train, y_train)

predict = pipe.predict(X_test)

print('With text process & TFITF')
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))


pipe1 = Pipeline([
    ('cv', CountVectorizer(analyzer=text_process)),
    ('naive', MultinomialNB())
])

pipe1.fit(X_train, y_train)

predict_pipe1 = pipe1.predict(X_test)

print('With Text process, but without TFITF')
print(confusion_matrix(y_test, predict_pipe1))
print('\n')
print(classification_report(y_test, predict_pipe1))

