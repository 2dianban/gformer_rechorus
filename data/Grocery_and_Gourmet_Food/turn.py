import os
import gzip
import subprocess 
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from scipy.sparse import coo_matrix

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

DATASET = 'Grocery_and_Gourmet_Food'
RAW_PATH = os.path.join('./', DATASET)
DATA_FILE = 'reviews_{}_5.json.gz'.format(DATASET)
META_FILE = 'meta_{}.json.gz'.format(DATASET)

RANDOM_SEED = 0
NEG_ITEMS = 99

if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)  

if not os.path.exists(os.path.join(RAW_PATH, DATA_FILE)):
    print('Downloading interaction data into ' + RAW_PATH)
    subprocess.call(
        f'cd {RAW_PATH} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{DATASET}_5.json.gz',
        shell=True
    )

if not os.path.exists(os.path.join(RAW_PATH, META_FILE)):
    print('Downloading item metadata into ' + RAW_PATH)
    subprocess.call(
        f'cd {RAW_PATH} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{DATASET}.json.gz',
        shell=True
    )

data_df = get_df(os.path.join(RAW_PATH, DATA_FILE))
meta_df = get_df(os.path.join(RAW_PATH, META_FILE))

useful_meta_df = meta_df[meta_df['asin'].isin(data_df['asin'])].reset_index(drop=True)

n_users = data_df['reviewerID'].value_counts().size
n_items = data_df['asin'].value_counts().size
n_clicks = len(data_df)
min_time = data_df['unixReviewTime'].min()
max_time = data_df['unixReviewTime'].max()
time_format = '%Y-%m-%d'

print('# Users:', n_users)
print('# Items:', n_items)
print('# Interactions:', n_clicks)
print('Time Span: {}/{}'.format(
    datetime.utcfromtimestamp(min_time).strftime(time_format),
    datetime.utcfromtimestamp(max_time).strftime(time_format))
)

np.random.seed(RANDOM_SEED)
out_df = data_df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id', 'unixReviewTime': 'time'})
out_df = out_df[['user_id', 'item_id', 'time']]
out_df = out_df.drop_duplicates(['user_id', 'item_id', 'time'])
out_df = out_df.sort_values(by=['time', 'user_id'], kind='mergesort').reset_index(drop=True)

uids = sorted(out_df['user_id'].unique())
user2id = dict(zip(uids, range(1, len(uids) + 1)))
iids = sorted(out_df['item_id'].unique())
item2id = dict(zip(iids, range(1, len(iids) + 1)))

out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])

clicked_item_set = dict()
for user_id, seq_df in out_df.groupby('user_id'):
    clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

def generate_dev_test(data_df):
    result_dfs = []
    n_items = data_df['item_id'].value_counts().size
    for idx in range(2):
        result_df = data_df.groupby('user_id').tail(1).copy()
        data_df = data_df.drop(result_df.index)
        neg_items = np.random.randint(1, n_items + 1, (len(result_df), NEG_ITEMS))
        for i, uid in enumerate(result_df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked:
                    neg_items[i][j] = np.random.randint(1, n_items + 1)
        result_df['neg_items'] = neg_items.tolist()
        result_dfs.append(result_df)
    return result_dfs, data_df

leave_df = out_df.groupby('user_id').head(1)
data_df = out_df.drop(leave_df.index)

[test_df, dev_df], data_df = generate_dev_test(data_df)
train_df = pd.concat([leave_df, data_df]).sort_index()

rows = train_df['user_id'].values - 1  
cols = train_df['item_id'].values - 1  
data = np.ones(len(train_df))  

rows_test = test_df['user_id'].values - 1 
cols_test = test_df['item_id'].values - 1 
data_test = np.ones(len(test_df))  

rows_val = dev_df['user_id'].values - 1 
cols_val = dev_df['item_id'].values - 1  
data_val = np.ones(len(dev_df))  

train_matrix = coo_matrix((data, (rows, cols)), shape=(train_df['user_id'].max(), train_df['item_id'].max()))
test_matrix = coo_matrix((data_test, (rows_test, cols_test)), shape=(test_df['user_id'].max(), test_df['item_id'].max()))
val_matrix = coo_matrix((data_val, (rows_val, cols_val)), shape=(dev_df['user_id'].max(), dev_df['item_id'].max()))

print(f'Training matrix shape: {train_matrix.shape}')
print(f'Training matrix non-zero entries: {train_matrix.nnz}')


with open(os.path.join(RAW_PATH, 'trnMat.pkl'), 'wb') as f:
    pickle.dump(train_matrix, f)
    
with open(os.path.join(RAW_PATH, 'tstMat.pkl'), 'wb') as f:
    pickle.dump(test_matrix, f)    

with open(os.path.join(RAW_PATH, 'valMat.pkl'), 'wb') as f:
    pickle.dump(val_matrix, f)

train_df.to_csv(os.path.join(RAW_PATH, 'train.csv'), sep='\t', index=False)
dev_df.to_csv(os.path.join(RAW_PATH, 'dev.csv'), sep='\t', index=False)
test_df.to_csv(os.path.join(RAW_PATH, 'test.csv'), sep='\t', index=False)
