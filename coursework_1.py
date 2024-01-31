# In order to run this python file it is necessary that the files from MovieLens20M from https://grouplens.org/datasets/movielens/20m/are in the same folder as this file

# Python file to do some preprocessing of the initial dataset (MovieLens20M)
# Note that more py files are being used for the preprocessing of the dataset.

# Import Necessary Libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle 

# there are the original 20M MovieLens datasets
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')
tags_users_df = pd.read_csv('tags.csv')
tags_movies_df = pd.read_csv('genome-scores.csv')
tags_id_df = pd.read_csv('genome-tags.csv')



#As we are using a deep model with a wide and deep architecture and view the problem as a classification one we run out of memory issues if we use the full datasets as we don't have enough CUDA memory to classify 25000 movies (the model has too many parameters). Also the training is extremely slow (for 10000 movies, 1 epoch is about 16 hours), therefore we reduce the dataset to 3000 movies, as it doesn't reduce the number of ratings substantially.

most_popular_movies = ratings_df['movieId'].value_counts()
most_popular_movies = most_popular_movies.index.tolist()[:3000]

ratings_df = ratings_df.loc[ratings_df['movieId'].isin(most_popular_movies), :]

# We still have about 17.5M ratings, so the dataset only slightly reduced.

# remove the movies we ignore from the other datasets too

tags_movies_df = tags_movies_df.loc[tags_movies_df['movieId'].isin(most_popular_movies), :]
tags_users_df = tags_users_df.loc[tags_users_df['movieId'].isin(most_popular_movies), :]
movies_df = movies_df.loc[movies_df['movieId'].isin(most_popular_movies), :]

# Genres column in movies_df contains genres in a weird format. Initially we one-hot-encode genres

movies_df['genres'] = movies_df['genres'].apply(lambda x:x.split('|'))
movies_df[['title','year']] = movies_df['title'].str.split('\((\d{4})\)',n=1, expand=True).drop(columns=2)
all_genres = []
for genre_list in movies_df['genres']:
    all_genres.extend(genre_list)
all_genres = list(set(all_genres))
# following https://medium.com/swlh/recommendation-system-for-movies-movielens-grouplens-171d30be334e we remove IMAX
all_genres.remove('IMAX') # this is not a genre of movies but a way of viewing movies

# create a new column for each genre and fill it with 1 if the movie has that genre
for genre in all_genres:
    movies_df[genre] = 0
    for index in range(movies_df.shape[0]):
        if genre in movies_df['genres'].iloc[index]:
            movies_df[genre].iloc[index] = 1
            
movies_df.rename(columns={"(no genres listed)": "No Genres"}, inplace=True)
movies_df.drop(columns=['genres'],inplace=True)

# Compute the number of unique users and movies

n_users = ratings_df['userId'].nunique()
n_movies = ratings_df['movieId'].nunique()
n_tags = tags_id_df['tagId'].nunique()

print(f"Number of unique users: {n_users}") 
print(f"Nuber of unique movies: {n_movies}")
print(f"Nuber of unique tags: {n_tags}")

# For the full dataset we had:
# Number of unique users: 138493  
# Nuber of unique movies: 26744   
# Nuber of unique tags: 1128

# We follow the idea from surprise package, where users and movies have raw id's (the ones in the original data) and inner id's (relabeled id's to input in our model).

unique_movies = ratings_df['movieId'].unique()
unique_movies.sort()
movie_dict = {raw_id: inner_id for inner_id, raw_id in enumerate(unique_movies)}
inner_to_raw_dict = {inner_id: raw_id for raw_id, inner_id in zip(list(movie_dict.keys()), list(movie_dict.values()))}

# to get back the raw user id we just add 1 to the inner id
user_files = [ratings_df,tags_users_df]
for file in user_files:
    file['userId'] = file['userId'].map(lambda x:(x-1))

movie_files = [ratings_df,movies_df,tags_movies_df, tags_users_df]
for file in movie_files:
    file['movieId'] = file['movieId'].map(movie_dict)

# to get back the raw movie id from the inner id we use: inner_to_raw_dict.get(innerId)

# Save the dictionary that decodes raw id from inner id of the movies, so that we can use it in different notebooks.

with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(inner_to_raw_dict, f)

# We also want all datasets to contain the same format of tags and them to start from 0 instead of 1. We also use the tags from users. Create a 'tag' column and fill it with 0 (user gave a tag to that movie) or 1 (user didn't give a tag to that movie)

tags_dictionary = {name:number for number, name in enumerate(tags_id_df['tag'].tolist(), start=1)}
number_to_tag_dict = {number-1:name for name, number in zip(list(tags_dictionary.keys()), tags_dictionary.values())}
tags_movies_df['tagId'] = tags_movies_df['tagId'].map(lambda x:(x-1))

# to get what a tag is when we have the tag id we use number_to_tag_dict.get(tagId)
tags_users_df['tag'] = 1
tags_users_df.drop_duplicates(['userId','movieId'],keep= 'first',inplace=True)
ratings_df = pd.merge(ratings_df, tags_users_df, how='outer', on=['userId','movieId'], suffixes = ('','_y'))
ratings_df['timestamp'] = ratings_df['timestamp'].fillna(ratings_df['timestamp_y'])
ratings_df.drop(columns=['timestamp_y'],inplace=True)
ratings_df['tag'] = ratings_df['tag'].fillna(0) # NA are the movies that are not tagged, so fill them with 0

del tags_id_df # empty some memory

# Also merging the tags from the movies with the movies data frame, as the dataset would be too large we do this in chunks (see below).

tags_movies_df = tags_movies_df.pivot_table(index='movieId',columns='tagId',values='relevance')
tags_movies_df = tags_movies_df.astype({col:'float32' for col in tags_movies_df.columns.tolist()})
tags_movies_df.to_csv("tags_movies_pivoted.csv") 
del tags_movies_df

# We consider the recommendation task, as one that we are in a point in time and want to use the information we have (up until that point in time) to predict new movies. Therefore we sort according to the timestamps, use the previous movies rated, tag's assigned, average rating and have as target the id of the next movie. Credit: https://www.kaggle.com/code/matanivanov/wide-deep-learning-for-recsys-with-pytorch

# Sort for each user, according to when they made their rating
ratings_df = ratings_df.sort_values(['userId', 'timestamp']).reset_index(drop=True)

# Change the data types so that we don't run into memory issues. Then merge the ratings and movies data frames

# as we have reduced the datasets, no longer movies contain the 'No Genres', genre
# all_genres[all_genres.index('(no genres listed)')] = 'No Genres' 
ratings_df = ratings_df.astype({'userId': 'int32',
                                'movieId': 'float32',
                                'rating':'float32',
                                'timestamp':'float32',
                                'tag':'float32'})
movies_df = movies_df.astype({'movieId':'float32',
                              'Crime':'int32',
                              'Comedy':'int32',
#                              'No Genres':'int32', this one was used for the full dataset
                              'Children':'int32',
                              'Action':'int32',
                              'Animation':'int32',
                              'Drama':'int32',
                              'Musical':'int32',
                              'Western':'int32',
                              'War':'int32',
                              'Documentary':'int32',
                              'Mystery':'int32',
                              'Horror':'int32',
                              'Romance':'int32',
                              'Adventure':'int32',
                              'Film-Noir':'int32',
                              'Thriller':'int32',
                              'Fantasy':'int32',
                              'Sci-Fi':'int32'})
ratings_df = ratings_df.merge(movies_df[['movieId','year'] + all_genres], on='movieId', how='left')

del movies_df
del tags_users_df

# This following few line are not necessary on the 5000/3000 movie dataset so we comment it out here

# We see that there are two movies we are still missing the year.
# We manualy check these movies
# 9877 is The Big Bang Theory - 2007
# 9878 is Fawlty Towers - 1975
# ratings_df.loc[ratings_df['movieId'] == 9877,'year'] = 2007
# ratings_df.loc[ratings_df['movieId'] == 9878,'year'] = 1975


# Turn the year column to integers starting from 0

ratings_df = ratings_df.astype({'year': 'int32'})
#ratings_df.eval('year=year-1890',inplace=True) #for full dataset
#ratings_df.eval('year=year-1902',inplace=True) #for 10000 movie dataset
ratings_df.eval('year=year-1920',inplace=True) #for 5000/3000 movie dataset
#ratings_df.eval('year=year-1927',inplace=True) #for 1000 dataset
ratings_df = ratings_df.astype({'year': 'int8'})

# Save the file and load it as chunks, so that we don't run into memory issues
ratings_df.to_csv("ratings_semi_processed.csv", index = False) 
del ratings_df
tags_pivoted = pd.read_csv("tags_movies_pivoted.csv")

# We choose the 50 tags with the highest genome scores to merge into the ratings dataframe.

tags_to_merge = tags_pivoted.describe().iloc[1].sort_values(ascending=False).iloc[:51].index
tags_pivoted_50 = tags_pivoted[tags_to_merge]
tags_pivoted_50.to_csv("tags_movies_pivoted_50.csv",header=True)
