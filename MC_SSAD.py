import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import copy
#import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
#from keras.optimizers import Adam
#from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def normalization(data,  option = 0):
    data = np.float32(data)
    if option == 0: # normalize data with respect to all bands, default
        maxValue = np.max(data)
        minValue = np.min(data)
        
        return (data - minValue)/(maxValue-minValue)
    elif option == 1: # normalize data with respect to individual bands
        for i in range(data.shape[2]):
            maxValue = np.max(data[:,:,i])
            minValue = np.min(data[:,:,i])
            
            data[:,:,i] = (data[:,:,i] - minValue) /(maxValue-minValue)
        
        return data
    
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, reg_pr):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=reg_pr)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x[:, -1, :]
    
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, seq_len, reg_pr):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=reg_pr)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, reg_pr):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, reg_pr)
        self.decoder = LSTMDecoder(hidden_dim, output_dim, seq_len, reg_pr)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def temporalize(X, lookback):
    output_X = []
    for i in range(len(X)-lookback+1):
        t = []
        for j in range(0,lookback):
            # Gather past records upto the lookback period
            t.append(X[[(i+j)], :])
        output_X.append(t)
    return output_X    

def trainAutoencoder(X, n_features, epochs = 300, hidden_dim = 12, batch_size = 32, timesteps = 2, l_rate = 0.001, reg_pr = 0.01):
    # Define the LSTM-based autoencoder
    # inputs = Input(shape=(timesteps, n_features))
    # encoded = LSTM(64, activation='relu', kernel_regularizer=l2(reg_pr))(inputs)
    # encoded = RepeatVector(timesteps)(encoded)
    # decoded = LSTM(64, return_sequences=True, activation='relu', kernel_regularizer=l2(reg_pr))(encoded)
    # decoded = TimeDistributed(Dense(n_features))(decoded)
    
    # autoencoder = Model(inputs, decoded)
    # autoencoder.compile(optimizer=Adam(learning_rate = l_rate), loss='mse')
    
    # # Train the autoencoder
    # history = autoencoder.fit(X, X, epochs=200, batch_size=32, verbose=0)
    
    # Model
    model = LSTMAutoencoder(n_features, hidden_dim, n_features, timesteps, reg_pr)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = l_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_X = torch.tensor(batch_X, dtype=torch.float32)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
        
    return model

def evaluate_model(model, test_data):
    
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        reconstructed_data = model(test_data)
    return reconstructed_data

def LSTMbasedAD(data, timesteps = 2, wout = 21, win = 11):
    [H,W,B] = data.shape #B = n_features
    data = normalization(data)
    
    data_test = np.float32(np.zeros((3*H,3*W,B)))
    data_test[H:2*H, W:2*W,:] = data[:,:,:]
    data_test[H:2*H, 0:W, :] = data[:, ::-1, :]
    data_test[H:2*H, 2*W:3*W, :] = data[:, ::-1, :]
    data_test[0:H, :, :] = data_test[2*H-1:H-1:-1, :, :]
    data_test[2*H:3*H, :, :] = data_test[2*H-1:H-1:-1, :, :]
    
    tout = np.int32(np.floor(wout/2))
    tin = np.int32(np.floor(win/2))
    result_mat = np.zeros((H + 2*tin, W + 2*tin))
    result_count = np.zeros((H + 2*tin, W + 2*tin))
    it = 0
    for i in range(H):
        for j in range(W):
            ii = i + H
            jj = j + W
            block_all = copy.deepcopy(data_test[ii-tout:ii+tout+1, jj-tout:jj+tout+1, :])
            block_test = copy.deepcopy(data_test[ii-tin:ii+tin+1, jj-tin:jj+tin+1, :])
            
            block_all[tout-tin:tout+tin+1, tout-tin:tout+tin+1, :] = np.nan
            block_all = block_all.reshape((wout*wout,B))
            block_all = block_all[~np.isnan(block_all[:, 0]), :]
            
            outer_data = temporalize(X = block_all, lookback = timesteps)
            outer_data = np.array(outer_data)
            outer_data = outer_data.reshape(outer_data.shape[0], timesteps, B)
            
            block_test = block_test.reshape((win*win,B))
            test_data = temporalize(X = block_test, lookback = timesteps)
            test_data = np.array(test_data)
            test_data = test_data.reshape(test_data.shape[0], timesteps, B)
            
            model = trainAutoencoder(outer_data, B)
            
            test_data = torch.tensor(test_data, dtype=torch.float32)
            reconstructions = evaluate_model(model, test_data)
            
            diff_sq = (test_data - reconstructions) ** 2
            diff_sq = diff_sq.detach().cpu().numpy()
            
            count = np.zeros((win*win))
            mat = np.zeros((win*win))
            for s in range(diff_sq.shape[0]):
                res = np.mean(diff_sq[s,:,:], axis=1)
                mat[s:s + timesteps] =  mat[s:s + timesteps] + res
                count[s:s + timesteps] =  count[s:s + timesteps] + 1
            
            mat = mat.reshape((win,win))
            count = count.reshape((win,win))
            result_mat[i:i+win, j:j+win] = result_mat[i:i+win, j:j+win] + mat 
            result_count[i:i+win, j:j+win] = result_count[i:i+win, j:j+win] + count 
            
            it = it + 1
            print(f"{it}. iteration completed.")
            
    final_result = result_mat / result_count
    return result_mat, result_count, final_result[tin:H+tin,tin:W+tin]


mat_c = sio.loadmat('pavia.mat')
data = mat_c['data']
gtruth = mat_c['groundtruth']

[H,W,B] = data.shape

[result_mat, result_count, final_result] = LSTMbasedAD(data)
# 
AUC_LSTM = roc_auc_score(np.reshape(gtruth,[H*W,1]), np.reshape(final_result,[H*W,1]))

plt.imshow(final_result, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(gtruth, cmap='hot', interpolation='nearest')
plt.show()


