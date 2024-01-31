# To run this file we need 'dataset/train_final.csv', 'dataset/test_final.csv','movie_dict.pkl' and 'movies.csv'. Note that 'movies.csv' is from the original MovieLens20M dataset.

# This file creates 'baseline_results.csv' which is used in the interface. It also provides results for different metrics on the baseline.
# This file in a real world environment should be re-run in order to update the csv file from which movies are recommender (for example once a day).

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
from tqdm import tqdm
import warnings

# We build a popularity recommender as a baseline. We will use the training data to find the most popular movies. Then recommend to the users in the testing data the top-n movies from the most popular ones, excluding the ones they have already seen.

ratings_train = pd.read_csv('dataset/train_final.csv', usecols=[0,1,2], names=['userId','movieId','rating'], dtype='float32')
ratings_test = pd.read_csv('dataset/test_final.csv', usecols=[0,1,2], names=['userId','movieId','rating'], dtype='float32')

# Replace the missing ratings (the ones we had a tag, but not a rating) with the average rating.
print('Mean rating:')
print(ratings_train[ratings_train['rating'] != -1]['rating'].mean())
# change the missing ratings with the value 3.5, the mean rating of the entire dataset

ratings_train.loc[ratings_train['rating'] == -1,'rating'] = 3.5
ratings_test.loc[ratings_test['rating'] == -1,'rating'] = 3.5

# We build a simple algorithm that produces the average rating of the movie

# First find the average rating for each movie
avg_rating = ratings_train.groupby('movieId')['rating'].mean().sort_values(ascending=False)

predictions = []
for i in range(ratings_test.shape[0]):
    try:
        movie = ratings_test.iloc[i,1]
        prediction = avg_rating.loc[movie]
    except:
        prediction = 0.5 # in case we haven't seen the movie in the training data
    predictions.append(prediction)
    
ratings_test['predictions'] = predictions

# Load the dictionary to turn inner_ids to raw_ids (for movies).
# Create a dictionary to turn raw_ids to movie titles.

with open('movie_dict.pkl', 'rb') as f:
    movie_dict = pickle.load(f)

movies_df = pd.read_csv('movies.csv')
id_to_title = {inner_id:title for inner_id,title in zip(movies_df['movieId'],movies_df['title'])}

# turn inner_ids to raw_ids
ratings_test['userId'] = ratings_test['userId'].map(lambda x:(x+1))
ratings_test['movieId'] = ratings_test['movieId'].map(movie_dict)

# use this if you want to get titles instead of ids
ratings_test['movieId'] = ratings_test['movieId'].map(id_to_title) 

# create a column with the real ranks of the ratings for each user
ratings_test['true_rank'] = ratings_test.groupby('userId')['rating'].rank(method='dense', ascending=False)



### Mean Squared Error ###
baseline_error = mean_squared_error(ratings_test['rating'], ratings_test['predictions'])
print(f"Baseline error: {baseline_error:.4f}")


### Mean Reciprocal Rank ###
# takes about 2-3 minutes
def mrr(predictions_df):
    """
    Functions that we insert a user and get the mean reciprocal rank, values in [0,1] the closer to 1 the better
    :param predictions_df: df that has user ids, movie ids/movie titles, ratings,rank of ratings and predictions 
    """
    predictions_df.sort_values(by=['userId','predictions'], ascending=False, inplace=True)
    
    rank_dict = {user:0 for user in predictions_df['userId'].unique()}
    reciprocal_ranks = [] # initialise an empty array to input the reciprocal rank of every user
    for user in tqdm(predictions_df['userId'].unique()): # for each user
        user_df = predictions_df[predictions_df['userId'] == user]
        rank = user_df['true_rank'].iloc[0] # get the rank of the highest prediction (the first recommendation)
        reciprocal_ranks.append(1/rank) # append the reciprocal of the rank to the array
        rank_dict[user] = rank
    return rank_dict, np.mean(reciprocal_ranks)

rank_dict, mean_rec_rank = mrr(ratings_test)
print(f"MRR: {mean_rec_rank:.4f}")


### Recommendations ###
def get_recommendations(user,predictions_df,movies=movies_df,n=5):
    """
    Function that we insert a user and return predictions
    :param user: raw user id
    :param predictions_df: df that has user ids, movie ids/movie titles and predictions
    :param n: number of recommendations
    """
    user_predictions = predictions_df[predictions_df['userId'] == user]
    user_predictions.sort_values(by=['userId', 'predictions'], ascending=False, inplace=True)
    recommend = (user_predictions[['movieId','predictions']].iloc[0:n]).reset_index(drop=True)
    if len(recommend) < n:
        print('Not enough info for this user, recommending less movies')
    return (user_predictions[['movieId','predictions']].iloc[0:n]).reset_index(drop=True)

print('Printing example recommendation')
print(get_recommendations(138493.0,predictions_df=ratings_test,n=5))


### Coverage ###
def coverage(predictions_df,total_movies=3000,n=5):
    """
    How many movies out of all the movies can the model recommend
    :param predictions_df: df movie ids/movie titles (and potentially more columns)
    :param n: number of recommendations
    """
    recommended_movies = np.array([])
    for user in tqdm(predictions_df['userId'].unique()):
        user_recommendations = get_recommendations(user,predictions_df,n)
        recommended_movies = np.append(recommended_movies, user_recommendations['movieId'].tolist())
        recommended_movies = np.unique(recommended_movies)
    return len(recommended_movies) / total_movies

# takes about 4 minutes
warnings.filterwarnings("ignore")
coverage_score = coverage(ratings_test)
warnings.filterwarnings("default")

print(f"{coverage_score*100:.2f}% of all the movies are being recommended")


### Save the resulting data frame ###
ratings_test.rename(columns={'movieId': 'title'},inplace=True)
ratings_test = pd.merge(ratings_test,movies_df[['title','genres']].drop_duplicates(subset=['title']),how='left',on='title')

ratings_test.to_csv('results/baseline_results.csv',index=False)
