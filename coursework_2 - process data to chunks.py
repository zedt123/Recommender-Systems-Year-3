# To run this file we need "ratings_semi_processed.csv" , "tags_movies_pivoted_50.csv", and the empty folders chunks_train,chunks_test with the subfolders data and hist. This file takes about 55 minutes and might crush if there is not enough RAM to read "ratings_semi_processed.csv"

# This file reads the semi processing ratings, processes it and saves it into smaller chunks (we need to process small chunks of it so that we don't run out of memory).

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# this file needs A LOT of ram to run

ratings_df = pd.read_csv("ratings_semi_processed.csv", dtype={'userId':'int32',
                                                              'movieId':'float32',
                                                              'rating':'float32',
                                                              'timestamp':'float32',
                                                              'tag':'float32',
                                                              'year':'int8',
                                                              'Crime':'int32',
                                                              'Comedy':'int32',
                                                              #'No Genres':'int32', 
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

tags_movies_pivoted = pd.read_csv('tags_movies_pivoted_50.csv',dtype='float32')

# In order to continue preprocess of the dataset we do it on chunks of the dataset

n_users = ratings_df['userId'].nunique()
print(f"We have {n_users} users.")

user_idxs = np.linspace(0,n_users-1,num=300).astype(int)

# Some preprocessing operations were taken from https://www.kaggle.com/code/matanivanov/wide-deep-learning-for-recsys-with-pytorch

#all_genres = ['Film-Noir','Fantasy','Animation','Documentary','Action','Romance','Adventure','Thriller','Musical','Mystery','Comedy','Drama','No Genres','Crime','War','Children','Sci-Fi','Horror']
all_genres = ['Film-Noir','Fantasy','Animation','Documentary','Action','Romance','Adventure','Thriller','Musical','Mystery','Comedy','Drama','Crime','War','Children','Sci-Fi','Horror']
cold_start_number = 5 # ignore the first ratings of some users
train_size = 0.8


def process(chunk,i):
    
    split = int(len(user_idxs) * 0.8)

    chunk['dummy_one'] = 1
    chunk['number_of_ratings'] = chunk.groupby('userId')['dummy_one'].cumsum()
    
    chunk['target'] = chunk.groupby('userId')['movieId'].shift(-1)
    
    chunk['avg_rating'] = chunk.groupby('userId')['rating'].cumsum() / chunk['number_of_ratings']
    
    chunk['history'] = chunk['movieId'].apply(lambda x: str(x)+" ")
    chunk['history'] = chunk.groupby('userId')['history'].apply(np.cumsum)
    chunk['history'] = chunk['history'].apply(lambda x: x.split()) # turns the numbers to a list    
    

    
    chunk.drop(columns=['dummy_one'],inplace=True)
    chunk['number_of_ratings'] = chunk['number_of_ratings'].astype('uint16')
    chunk['avg_rating'] = chunk['avg_rating'].astype('float32')
    
    chunk.index = chunk.index.astype('uint32')
    
    # average rating for each genre, share of total ratings for each genre
    for genre in all_genres:
        chunk[f'{genre}_avg_rating'] = chunk[genre]*chunk['rating']
        chunk[genre] = chunk.groupby('userId')[genre].cumsum()
        chunk[f'{genre}_avg_rating'] = chunk.groupby('userId')[f'{genre}_avg_rating'].cumsum() / chunk[genre]
        chunk[f'{genre}_avg_rating'] = chunk[f'{genre}_avg_rating'].fillna(0)
        chunk[genre] = chunk[genre] / chunk['number_of_ratings']
        
        chunk[f'{genre}_avg_rating'] = chunk[f'{genre}_avg_rating'].astype('float32')
        chunk[genre] = chunk[genre].astype('float32')
        
    chunk = pd.merge(chunk, tags_movies_pivoted, how='left', left_on=['movieId'], right_on=['movieId'])
    chunk = chunk[~chunk['target'].isna()]
    chunk.fillna(-1,inplace=True)
    chunk = chunk[chunk['number_of_ratings'] >= cold_start_number]    
    history = chunk['history']  
    
    if i<=split:
        chunk.drop(columns=['history']).to_csv('chunks_train/data/chunk_' + str(i) + '.csv', index=False, header=True)
        history.to_csv('chunks_train/hist/hist_' + str(i) + '.csv', index=False, header=False)
    else:
        chunk.drop(columns=['history']).to_csv('chunks_test/data/chunk_' + str(i) + '.csv', index=False, header=True)
        history.to_csv('chunks_test/hist/hist_' + str(i) + '.csv', index=False, header=False)
    

# For this to run make sure there exist a folder chunks_train  and chunks_test with subfolders data and history
# takes about 50 minutes to run

warnings.filterwarnings("ignore")

index = ratings_df.index[ratings_df['userId'] == user_idxs[1]][-1]
chunk = ratings_df.iloc[0:index+1]
process(chunk,0)

#index = ratings_df.index[ratings_df['userId'] == user_idxs[150]][-1] # i+1
# want i+1 = 150 --> i=149 -->start 150
for i in tqdm(range(0,len(user_idxs)-2)): # i
    user = user_idxs[i+2]
    next_index = ratings_df.index[ratings_df['userId'] == user][-1]
    chunk = ratings_df.iloc[index+1:next_index+1]
    process(chunk,i+1)
    index = next_index
    
warnings.filterwarnings("default")

# The Unnamed:0 column is the movieIds duplicated














































































































































































