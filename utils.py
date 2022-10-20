import pandas as pd
import numpy as np
import torch
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    #pl.seed_everything(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(size):
    train_set = pd.read_csv('train_data.csv',iterator=True,chunksize=size)
    train_set = train_set.get_chunk(size)
    #train_set = pd.read_feather('train_data.ftr',use_threads=True)
    train_lab = pd.read_csv('train_labels.csv')
    train_set = pd.merge(left=train_set,right=train_lab,on='customer_ID')
    return train_set

from numpy import float64 
from sklearn.preprocessing import OneHotEncoder
def preprocess(dataset):
    cus = dataset.groupby('customer_ID').count()[['target']]
    #cus = cus
    dataset = pd.merge(left=dataset,right=cus,left_on='customer_ID',right_on=cus.index)
    dataset = dataset[dataset['target_y'] == 13]
    dataset['D_87'].fillna(0,inplace=True)
    def categori(dataset):
        cate = dataset[['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']]
        dataset = dataset.drop(columns=['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'])
        encoder = OneHotEncoder(dtype= float64)
        oh = encoder.fit_transform(cate).toarray()
        one_hot = [f'category{i}' for i in range(len(oh[0]))]
    
        dataset[one_hot] = oh
        return dataset
    dataset = categori(dataset)
    dataset.fillna(method='pad',inplace=True)
    dataset.fillna(value=1e-8,inplace=True)
    dataset['S_2'] = pd.to_datetime(dataset['S_2'])
    return dataset

from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, x_length = 13,y_length=1):
        self.features = features
        self.target = target
        self.x_length = x_length
        self.y_length = y_length
        self.X = torch.tensor(dataframe[features].values)
        self.Y = torch.tensor(dataframe[target].values)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        j = i
        if j >= self.y_length - 1:
            j_start = j - self.y_length + 1
            y = self.Y[j_start:(j + 1), :]
        else:
            padding_y = self.Y[0].repeat(self.y_length - j - 1, 1)
            y = self.Y[0:(j + 1), :]
            y = torch.cat((padding_y, y), 0)

        if i >= self.x_length - 1:
            i_start = i - self.x_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding_x = self.X[0].repeat(self.x_length - i - 1, 1)

            x = self.X[0:(i + 1), :]
            x = torch.cat((padding_x, x), 0)

        return x, y