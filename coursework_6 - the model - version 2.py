import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pickle
import warnings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://medium.com/swlh/how-to-use-pytorch-dataloaders-to-work-with-enormously-large-text-files-bbd672e955a0

class DataIterableDataset(IterableDataset):
    def __init__(self,file,history):
        self.file = file
        self.history = history
    
    def line_mapper(self,line):
        array =  np.array(line.split(','))
        target = array[2].astype('float32') # 25 is target
        features = np.delete(array,[0,1,2,3,24,25,44]).astype('float32') # 24 is number of ratings, 0-1-2 userId,movieId,rating, 44 is movieId duplicated
        return features, target
    
    def hist_line_mapper(self,line):
        line = line.strip('\n')
        line = line.replace('\'','')
        line = line.strip('"')
        line = line.replace('[','')
        line = line.replace(']','')        
        return line
    
    def __iter__(self):
        iterator = open(self.file)
        hist_iterator = open(self.history)
        mapped_itr = map(self.line_mapper, iterator)
        mapped_hist = map(self.hist_line_mapper, hist_iterator)
        
        ziped_iterator = zip(mapped_itr, mapped_hist)
        
        return ziped_iterator

train_data = DataIterableDataset('dataset/train_final.csv', 'dataset/train_hist.csv')
train_iterable = DataLoader(train_data, batch_size=64)

# https://www.kaggle.com/code/matanivanov/wide-deep-learning-for-recsys-with-pytorch
# Number of maximum movies in history for different dataset reductions.
# full - 26744
# 10000 - 6563
# 5000 - 4151
# 3000 - 2834
# 1000 - 998
class WDModel(nn.Module):
    def __init__(self,n_features=88,n_movies = 3000,max_hist_movies = 2834,n_embedding=16):
        super().__init__()
        self.emb = nn.Embedding(n_movies,n_embedding,padding_idx=0)
        self.b_norm = nn.BatchNorm1d(n_features) 
        self.layer = nn.Sequential(nn.Linear(n_features + n_embedding,1024),
                                   nn.ReLU(),
                                   nn.Linear(1024,512),
                                   nn.ReLU(),
                                   nn.Linear(512,256),
                                   nn.ReLU(),                                   
        )
        self.out = nn.Linear(n_movies + 256,10) # classify ratings as 0.5,1.0,1.5...
        
    def forward(self,features,h_sparse,h_idx):
        emb_idx = self.emb(h_idx) # shape: [batch_size,n_movies,n_embedding]
        emb_idx = torch.mean(emb_idx.to(torch.float32),dim=1) # shape: [batch_size,n_embedding]
        
        features = self.b_norm(features)
        
        x = torch.cat((features,emb_idx),dim=1) # shape: [batch_size, n_features + n_embedding]
        x = self.layer(x).to(torch.float64) # shape: [batch_size,256]
        
        x = torch.cat((x,h_sparse),dim=1).to(torch.float32) # shape: [batch_size, 256 + n_movies]
        x = self.out(x) # shape: [batch_size,10]
        return x


def h_process(history, n_movies = 3000,max_hist_movies = 2834):
    first = True
    
    for h in history:
        h = np.array(h.split(',')).astype(float).astype(int)

        h_sparse = np.zeros(n_movies)
        h_sparse[h] = 1 # fill with 1 if the movie was rated
        h_sparse = torch.tensor([h_sparse]) # turn to tensor

        h_idx = np.pad(h,(0, (max_hist_movies - len(h))), 'constant') # pad with 0
        h_idx = torch.tensor([h_idx])
        if first:
            h_sparse_tensor = h_sparse
            h_idx_tensor = h_idx
            first = False
        else:
            h_sparse_tensor = torch.cat((h_sparse_tensor,h_sparse))
            h_idx_tensor = torch.cat((h_idx_tensor,h_idx))  
    return h_sparse_tensor,h_idx_tensor


model = WDModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# commend to NOT load the model
# """
if device == 'cpu':
    params = torch.load('wnd_2.chkpt',map_location=torch.device('cpu'))
else:
    params = torch.load('wnd_2.chkpt')
model.load_state_dict(params['model'])
optimizer.load_state_dict(params['optimiser'])
epoch = params['epoch']
# """

# commend to train the model (e.g. if we are loading weights and want to just get predictions)
"""
# 3 epochs were about 14 hours
epochs = 3
model.train()

for epoch in range(epochs):
    
    running_loss = 0
    for i, batch in enumerate(train_iterable):
        data,history = batch
        features,target = data
        target[target == -1] = 3.5
        target = (target * 2) - 1 # turn it from the range (0.5-5.0) to (0-9)
        target = target.to(torch.int64)
        
        h_sparse_tensor,h_idx_tensor = h_process(history)
        
        features = features.to(device)
        target = target.to(device)
        h_sparse_tensor,h_idx_tensor = h_sparse_tensor.to(device),h_idx_tensor.to(device)
        
        optimizer.zero_grad()
            
        x = model(features,h_sparse_tensor,h_idx_tensor)

        # need to insert the predictions and the true rating
        loss = criterion(x,target)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i%10000 == 0:
            print(f"Epoch: {epoch} | Iteration: {i} | Loss: {running_loss/10000:.4f}")
            running_loss = 0
"""

### Save the Model ###

torch.save({'model':model.state_dict(), 'optimiser':optimizer.state_dict(), 'epoch':epoch}, 'wnd_2.chkpt')


### Evaluate the Model ###

test_data = DataIterableDataset('dataset/test_final.csv', 'dataset/test_hist.csv')
test_iterable = DataLoader(test_data, batch_size=64)

# We need to evaluate the model the SAME way as we evaluated the baseline
# The way we evaluated the baseline is: for every user/movie pair in the testing dataset we gave it a rating between 0.5 and 5
batch_size = 64
predictions = np.array([])
ratings_test = ratings_test = pd.read_csv('dataset/test_final.csv', usecols=[0,1,2], names=['userId','movieId','rating'], dtype='float32')


model.eval()
with torch.inference_mode():
    for i, batch in enumerate(test_iterable):
        data,history = batch
        features,target = data
        
        h_sparse_tensor,h_idx_tensor = h_process(history)
        
        features = features.to(device)
        h_sparse_tensor,h_idx_tensor = h_sparse_tensor.to(device),h_idx_tensor.to(device) 
        
        x = model(features,h_sparse_tensor,h_idx_tensor) # there are the logits, shape:[batch_size,10]
        preds = ((torch.argmax(x,dim=1).cpu()).numpy() + 1.0) / 2.0

        predictions = np.append(predictions,preds)
        
        if i%10000==0:
            print(i)

ratings_test['predictions'] = predictions


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
print(f"Mean squared error: {baseline_error}")


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
print(f"MRR: {mean_rec_rank}")


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

print("Example recommendation:")
print(get_recommendations(138493.0,predictions_df=ratings_test,n=5))


### Coverage ###

def coverage(predictions_df,total_movies=5000,n=5):
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


### Save the Resulting Data Frame ###

ratings_test.rename(columns={'movieId': 'title'},inplace=True)
ratings_test = pd.merge(ratings_test,movies_df[['title','genres']].drop_duplicates(subset=['title']),how='left',on='title')

ratings_test.to_csv('results/wnd_2.csv',index=False)