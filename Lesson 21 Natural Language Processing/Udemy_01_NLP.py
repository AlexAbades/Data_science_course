import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# nltk has a lot of data and a lot of packages, in order to get these packages, we can call download_shell and serch
# for that one which interest us, if not, you can only call .download and download all


# nltk.download_shell()


messages = [line.rstrip() for line in open(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\20"
                                           r"-Natural-Language-Processing\smsspamcollection\SMSSpamCollection")]
print(len(messages))
print('\n')
print(messages[50])
print('\n')

for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')

print(messages[0])

# We can see that it's a:  tab separated values file, or TSV where the first column it's a label and the second is
# the message itself

messages = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\20-Natural-Language"
                       r"-Processing\smsspamcollection\SMSSpamCollection", sep='\t', names=['label', 'message'])
print(messages.head())

print(messages.describe())

print('\n')

print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)
print(messages)

sns.set_style('whitegrid')
messages['length'].hist(bins=200)
plt.show()

# There is some very long messages
print(messages['length'].describe())

print(messages[messages['length'] == messages['length'].max()]['message'].iloc[0])

messages.hist(column='length', by='label', bins=60, figsize=(12, 4))
plt.show()

import string

# We want to remove the punctuation

mess = 'Sample message! Notice: it has punctuation.'

print(string.punctuation)

# We can use list comprehension in order to pass in for every character
nopunc = [c for c in mess if c not in string.punctuation]

print(nopunc)
# remove the punctuations
print('\n')

# If we want to join elements in  a list together
print('Example')
x = 'a b c d'.split()
print(x)
x1 = '+++'.join(x)
print(x1)
x2 = 'H5'.join(x)
print(x2)
x3 = ''.join(x)
print(x3)

nopunc = ''.join(nopunc)
print(nopunc)

from nltk.corpus import stopwords

# They're very common words that it can tell us anything about the differences text

print(stopwords.words('english'))

nopunc = nopunc.split()
# We split it again to have a list, and then we can remove the stopwords

clean_mess = [word for word in nopunc if word.lower() not in stopwords.words('english')]

print(clean_mess)
print('\n')


def text_process(mess):
    """
    :param mess: A message which we want to clear
    :return: The message cleaned

    1st: We want to remove the punctuation
    2nd: remove stop words
    3th: Resturn list of clean text words
    """
    nonpunc = [char for char in mess if char not in string.punctuation]
    nonpunc = ''.join(nonpunc)
    clean = [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
    return clean

print(text_process(mess))

# TOKEN: The word that we actually want.
# TOKENIZE: The process that we just did, remove the punctuation and words
# that we are not interested and get the words that we want (TOKEN)

print(messages.head())
print('\n')

print(messages['message'].head().apply(text_process))

# We could continue to normalize our Data, nltk library has a lot of tools and documentation for other methods. For
# example Stemming it's a really common way to continue pre processing text data, and what it basically does it's,
# if in our text we have a bunch of similar words, such as running run ran, as these words are basically telling you
# the same, stemming it tries to brake down these words and return run. We need a dictionary
# As our data set has a lot of short hand, it's not useful

# We have now to convert our messages into a vector that our machine learning understand

# 1st: We are going to count how many times does a word occur in each message (Known as a term frequency)
# 2nd: We are going to weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 3th: Normalize the vectors to until length, to abstract from the original text length (L2 norm)

# What we are going to do know with sklearn, it's to convert into a matrix where the columns ara going to be the fully
# messages and the rows are going to be words, where the columns it has the total number of columns, the rows are going
# to have the total number og words.
# This is going to result in a matrix with a large number of 0, so what sklearn does it's give the result in a
# Sparse Matrix

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

print(bow_transformer)

print(len(bow_transformer.vocabulary_))

# Now we can take one sample text message and get its bag of words count as a vector putting to use our bag transformer
# bow_transformer

mess4 = messages['message'][3]
print(mess4)

bow4 = bow_transformer.transform([mess4])

print(bow4)

print('\n')

print(bow4.shape)

print('\n')
# To get specific words from the bag we can pass as an index, so if we want to know which word appears twice in the
# message, we pass the index of that word into our bag feature names

print(bow_transformer.get_feature_names()[4068])
print('\n')
print(bow_transformer.get_feature_names()[9554])
print('\n')
# Now we can repeat the process with all the messages

messages_bow = bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)

# We can check the amount of non zero occurrences by just calling the object (messages_bow) with the .nnz

print(messages_bow.nnz)

# Check the sparsity: comparing the number of non zero messages versus the actual number of messages

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

print('Sparsity : {}'.format(sparsity))

# Now the term weight and normalization can be done with TF.IDF ( term frequency inverse normalization

from sklearn.feature_extraction.text import TfidfTransformer

# So now we can create an instance

tfidf_transform = TfidfTransformer().fit(messages_bow)

# What itfidf it's going to evaluate how important it's a word for a collection of documents. The importance increases
# proportionally to the number of the times a word appears in the document but it's offset by the frequency of the word
# in the collection.
# Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by
# the number of documents where the specific term appears.

# IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are
# considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a
# lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare
# ones, by computing the following:
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

# Consider a document containing 100 words wherein the word cat appears 3 times.
#
# The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the
# word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as
# log( 10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.

tfidf4 = tfidf_transform.transform(bow4)

print(tfidf4)

# we can grab the tfidf from a specific word from the bag of words, such as

print(tfidf_transform.idf_[bow_transformer.vocabulary_['university']])

# Now we are going to convert the entire bag of words corpus into a TF.IDF corpus at once

messages_tfidf = tfidf_transform.transform(messages_bow)

# Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

spam_detect_model.predict(tfidf4)[0]

print(messages['label'][3])

# To run the naive bayes theorem we can run into all the data base

all_pred = spam_detect_model.predict(messages_tfidf)

# In real cases we should have to make the train test split

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)

print(msg_train)

# Sklearn has a pipeline that allow us to skip all this process.

from sklearn.pipeline import Pipeline

# We create an object, and it's going to be an instance of Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # String to tokens,integer step
    ('tfidf', TfidfTransformer()),  # We make the tfitf transformation
    ('classifier', MultinomialNB())  # Then we call to our model classifier

])

# You can treat the pipline and treat it as a normal estimator

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(label_test, predictions))

# We can change the classifier method as the Random Forest or something like that
