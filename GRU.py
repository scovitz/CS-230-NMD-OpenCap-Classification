#https://www.kaggle.com/code/malekzadeharman/rnn-gated-recurrent-unit/notebook

# Configuration
import pandas as pd
from collections import Counter
import spacy
import numpy as np
import re
import string
import gensim
import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import spacy
import pickle as pkl #Automatically in the python environment
import requests

config = {
    'data_path': 'data.csv',
    'batch_size': 64,
    'device': 'cuda', # mps for mac m1, cuda for gpu-enabled devices
    'learning_rate': 0.01,
    'num_epochs': 100,
    'train_size': 0.8,
    'random_seed': 50
}

np.random.seed(config['random_seed'])
torch.random.manual_seed(config['random_seed'])

# Load the dataset, this is currently Amazon's stockmarket data, change this to our data
url = 'https://www.dropbox.com/scl/fi/5zgutd3y6sm5jwuak60rp/data.csv?rlkey=2mivltwxvmx3rtjfzhltp0e09&dl=1'
response = requests.get(url)

# Save the file as 'data.csv'
with open('data.csv', 'wb') as file:
    file.write(response.content)

df = pd.read_csv(config['data_path'])
df

# Data Preprocessing
df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
df

dates = df['Date'].values
close_prices = df['Close'].values

data_dict = {}
for idx, date in enumerate(dates[7:], start=7):
    data_dict[date] = {
            'target': close_prices[idx],
            't-1': close_prices[idx-1],
            't-2': close_prices[idx-2],
            't-3': close_prices[idx-3],
            't-4': close_prices[idx-4],
            't-5': close_prices[idx-5],
            't-6': close_prices[idx-6],
            't-7': close_prices[idx-7],
        }
df = pd.DataFrame.from_dict(data_dict, orient='index')

df = df[['t-7', 't-6', 't-5', 't-4', 't-3', 't-2', 't-1', 'target']]
df

scaler = MinMaxScaler(feature_range=(-1, 1))
df[['t-7', 't-6', 't-5', 't-4', 't-3', 't-2', 't-1', 'target']] = scaler.fit_transform(df[['t-7', 't-6', 't-5', 't-4', 't-3', 't-2', 't-1', 'target']])
df

x = df[['t-7', 't-6', 't-5', 't-4', 't-3', 't-2', 't-1']].values
y = df['target'].values

# Splitting the dataset into test and training sets

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=config['train_size'], shuffle=True)
x_test, x_val, y_test, y_val = train_test_split(x_rem, y_rem, train_size=0.5, shuffle=True)

# Make a sliding window of size 7 over the dataset

class StockDataset(Dataset):
    def __init__(self, values, targets):
        self.values = values.reshape(-1, 7, 1)
        self.labels = np.array(targets).reshape(-1, 1)
    def __len__(self):
        return self.values.shape[0]
    def __getitem__(self, idx):        
        return Tensor(self.values[idx]).to(config['device']), Tensor(self.labels[idx]).to(config['device'])

train_dataset = StockDataset(x_train, y_train)
test_dataset = StockDataset(x_test, y_test)
val_dataset = StockDataset(x_val, y_val)

# To iterate over the data we need to create a data loader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

# Dropout layer for regularization

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=4, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(4, 1)
    
    def forward(self, x):
        output, _ = self.gru(x) # output shape: (batch_size, seq_len, hidden_size)
        output = self.dropout(output)
        output = self.dense(output[:, -1, :]) # output shape: (batch_size, 1)
        return output

model = Model().to(config['device'])

model(torch.randn(32, 7, 1).to(config['device'])).shape

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

criterion = nn.MSELoss()

def train_loop(dataloader, model, loss_fn, optimizer, epoch_num):
    num_points = len(dataloader.dataset)
    for batch, (features, labels) in enumerate(dataloader):        
        # Compute prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)
        
        # Backpropagation
        optimizer.zero_grad() # sets gradients of all model parameters to zero
        loss.backward() # calculate the gradients again
        optimizer.step() # w = w - learning_rate * grad(loss)_with_respect_to_w

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(features)
            print(f"\r Epoch {epoch_num} - loss: {loss:>7f}  [{current:>5d}/{num_points:>5d}]", end=" ")

def test_loop(dataloader, model, loss_fn, epoch_num, name):
    num_points = len(dataloader.dataset)
    sum_test_loss = 0

    with torch.no_grad():
        for batch, (features, labels) in enumerate(dataloader):
            pred = model(features)
            sum_test_loss += loss_fn(pred, labels).item() # add the current loss to the sum of the losses
            
    sum_test_loss /= num_points
    print(f"\r Epoch {epoch_num} - {name} Avg loss: {sum_test_loss:>8f}", end=" ")

for epoch_num in range(1, config['num_epochs']+1):
    train_loop(train_loader, model, criterion, optimizer, epoch_num)
    test_loop(val_loader, model, criterion, epoch_num, 'Development/Validation')

test_loop(test_loader, model, criterion, epoch_num, 'Test')

predictions = model(Tensor(x_test.reshape(-1, 7, 1)).to(config['device'])).cpu().detach().numpy().squeeze()

true_values = y_test[:]

r2_score(true_values, predictions)

pkl.dump(scaler, open('scaler.pkl', 'wb'))  # Saves to the current working directory

scaler = pkl.load(open('scaler.pkl', 'rb'))

concatenated_values = np.concatenate((x_test[0].reshape(1, -1), predictions[0].reshape(1, -1)), axis=1)

scaler.inverse_transform(concatenated_values) # the last value is the predicted value

def predict_next_day(previous_days):
    raw_values = np.concatenate((previous_days, [0]), axis=0).reshape(1, -1)
    scaled_values = scaler.transform(raw_values)
    scaled_input = scaled_values.squeeze()[:-1].reshape(-1, 7, 1)
    output = model(Tensor(scaled_input).to(config['device'])).cpu().detach().numpy()
    scaled_values = np.concatenate((previous_days.reshape(1,-1), output), axis=1)
    raw_values = scaler.inverse_transform(scaled_values)
    prediction = raw_values.squeeze()[-1]
    return prediction
    

predict_next_day(x_test[0])