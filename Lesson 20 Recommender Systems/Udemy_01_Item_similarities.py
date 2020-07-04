import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

columns_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\19-Recommender-Systems\u.data",
                 sep='\t', names=columns_names)

print(df.head())

movie_titles = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\19-Recommender-Systems"
                           r"\Movie_Id_Titles")
print(movie_titles.head())

df = pd.merge(df, movie_titles, on='item_id')
print(df.head())

print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
# As we made a groupby, it could be that for some films, we have only a rate of 1 person and it it could be a 5

print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

print(ratings.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

print(ratings.head())

ratings['num of ratings'].hist(bins=70)
plt.show()

ratings['rating'].hist(bins=70)
plt.show()

sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)
plt.show()

# Building a recommendation system

# matrix,

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat)

print(ratings.sort_values('num of ratings', ascending=False).head(10))

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

print(starwars_user_ratings)

# Compute pairwise correlation between rows or columns of Two DataFrame objects, so we can see the correlation of the
# movies by the user rating

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)

print(corr_starwars)

print(corr_starwars.sort_values('Correlation', ascending=False).head(10))

# The problems here are that a 1 correlation doesn't make sense, because most likely these movies happened to have
# been seen only by one person that also happened to rate Star Wars five stars

# We can fix these filtering out movies that have less than a certain number of reviews so we can set a threshold for
# the number of ratings necessary in order to be put into our model

corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars)
# We can use joint instead of merge, because we have the same index in the two dataframe

print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False).head())

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar = corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending=False)
print(corr_liarliar.head(10))
