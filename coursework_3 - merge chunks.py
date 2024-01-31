# We need the chunk files to merge them. We already have the merged files so there is no need to run this file.

# We first merge the chunks. We end up with four (4) datasets; the training and testing features, the training and testing rating history.

from os import listdir
import re
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

train_files = [csv for csv in listdir('chunks_train/data')]
train_hist = [csv for csv in listdir('chunks_train/hist')]
test_files = [csv for csv in listdir('chunks_test/data')]
test_hist = [csv for csv in listdir('chunks_test/hist')] 

try:
    train_files.remove('.ipynb_checkpoints')
except:
    pass
try:
    train_hist.remove('.ipynb_checkpoints')
except:
    pass    
try:
    test_files.remove('.ipynb_checkpoints')
except:
    pass    
try:
    test_hist.remove('.ipynb_checkpoints')
except:
    pass    

# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


train_files.sort(key=natural_keys)
train_hist.sort(key=natural_keys)
test_files.sort(key=natural_keys)
test_hist.sort(key=natural_keys)

# make sure the directory dataset exists

with open('dataset/test_final.csv','w') as f1:
    for file in tqdm(test_files):
        with open('chunks_test/data/'+file,'r') as f2:
            next(f2)
            for line in f2:
                f1.write(line)

with open('dataset/train_final.csv','w') as f1:
    for file in tqdm(train_files):
        with open('chunks_train/data/'+file,'r') as f2:
            next(f2)
            for line in f2:
                f1.write(line)


# Merge the chunks

def my_merge_hist(dataset,data_folder,path):
    for csv in tqdm(dataset):
        pd.read_csv(data_folder + csv).to_csv(path, mode="a", index=False)   
        
my_merge_hist(train_hist,'chunks_train/hist/','dataset/train_hist.csv')
my_merge_hist(test_hist,'chunks_test/hist/','dataset/test_hist.csv')

# Find the largest number of movies in history (we need this for the model)
n = pd.read_csv('dataset/train_final.csv', usecols=[24], names = ['n_ratings'])
print(n.max())
m = pd.read_csv('dataset/test_final.csv', usecols=[24], names = ['n_ratings'])
print(m.max())





