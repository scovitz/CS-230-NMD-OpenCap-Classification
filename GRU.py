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
import csv

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


# Save the file as 'data.csv'
with open('class_info.csv', 'r') as file:

    csv_reader = csv.reader(file)

    for row in csv_reader:
        print(row)

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('class_info.csv')
Class = df['Class'].values
ID = df['ID'].values # Don't put into the model

df2 = pd.read_csv('video_features.csv')
mwrt_ankle_elev = df2['10mwrt_ankle_elev'].values
mwrt_com_sway = df2['10mwrt_com_sway'].values
mwrt_mean_max_ka = df2['10mwrt_mean_max_ka'].values
mwrt_mean_ptp_hip_add = df2['10mwrt_mean_ptp_hip_add'].values
mwrt_speed = df2['10mwrt_speed'].values
mwrt_stride_len = df2['10mwrt_stride_len'].values
mwrt_stride_time = df2['10mwrt_stride_time'].values
mwrt_trunk_lean = df2['10mwrt_trunk_lean'].values
mwt_ankle_elev = df2['10mwt_ankle_elev'].values
mwt_com_sway = df2['10mwt_com_sway'].values
mwt_mean_max_ka = df2['10mwt_mean_max_ka'].values
mwt_mean_ptp_hip_add = df2['10mwt_mean_ptp_hip_add'].values
mwt_speed = df2['10mwt_speed'].values
mwt_stride_len = df2['10mwt_stride_len'].values
mwt_stride_time = df2['10mwt_stride_time'].values
mwt_trunk_lean = df2['10mwt_trunk_lean'].values
xsts_lean_max = df2['5xsts_lean_max'].values
xsts_stance_width = df2['5xsts_stance_width'].values
xsts_time_5 = df2['5xsts_time_5'].values
arm_rom_rw_area = df2['arm_rom_rw_area'].values
brooke_max_ea_at_max_min_sa = df2['brooke_max_ea_at_max_min_sa'].values
brooke_max_mean_sa = df2['brooke_max_mean_sa'].values
brooke_max_min_sa = df2['brooke_max_min_sa'].values
brooke_max_sa_ea_ratio = df2['brooke_max_sa_ea_ratio'].values
curls_max_mean_ea = df2['curls_max_mean_ea'].values
curls_min_max_ea = df2['curls_min_max_ea'].values
jump_max_com_vel = df2['jump_max_com_vel'].values
toe_stand_int_com_elev = df2['toe_stand_int_com_elev'].values
toe_stand_int_mean_heel_elev = df2['toe_stand_int_mean_heel_elev'].values
toe_stand_int_trunk_lean = df2['toe_stand_int_trunk_lean'].values
toe_stand_mean_int_aa = df2['toe_stand_mean_int_aa'].values
tug_cone_time = df2['tug_cone_time'].values
tug_cone_turn_avel = df2['tug_cone_turn_avel'].values
tug_cone_turn_max_avel = df2['tug_cone_turn_max_avel'].values

x = df2[['10mwrt_ankle_elev', '10mwrt_com_sway', '10mwrt_mean_max_ka', '10mwrt_mean_ptp_hip_add', '10mwrt_speed', '10mwrt_stride_len', '10mwrt_stride_time']].values
y = df['Class'].values

# Splitting the dataset into test and training sets

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=config['train_size'], shuffle=True)
x_test, x_val, y_test, y_val = train_test_split(x_rem, y_rem, train_size=0.5, shuffle=True)

# Make a sliding window of size Not 7 but 1 over the dataset

class StockDataset(Dataset):
    def __init__(self, values, targets):
        self.values = values.reshape(-1, 1, 1)
        self.labels = np.array(targets).reshape(-1, 1)
    def __len__(self):
        return self.values.shape[0]
    def __getitem__(self, idx):        
        return Tensor(self.values[idx]).to(config['device']), Tensor(self.labels[idx]).to(config['device'])

train_dataset = StockDataset(x_train, y_train)
test_dataset = StockDataset(x_test, y_test)
val_dataset = StockDataset(x_val, y_val)

# To iterate over the data we need to create a data loader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

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

model(torch.randn(32, 1, 1).to(config['device'])).shape

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

criterion = nn.BCEWithLogitsLoss() #Appropriate for binary classification

def train_loop(dataloader, model, loss_fn, optimizer, epoch_num):
    num_points = len(dataloader.dataset)
    for batch, (features, labels) in enumerate(dataloader):
        print(f"Batch {batch}, Features shape: {features.shape}")
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
    scaled_input = scaled_values.squeeze()[:-1].reshape(-1, 1, 1)
    output = model(Tensor(scaled_input).to(config['device'])).cpu().detach().numpy()
    scaled_values = np.concatenate((previous_days.reshape(1,-1), output), axis=1)
    raw_values = scaler.inverse_transform(scaled_values)
    prediction = raw_values.squeeze()[-1]
    return prediction
    

predict_next_day(x_test[0])