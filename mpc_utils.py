import pandas as pd
import os
import numpy as np
import torch
import numpy as np

from models.lmu_torch import LMUModel


def load_data(folder, include_L=False):
    X = []
    Y = []
    for fname in sorted(os.listdir(folder)):
        df = pd.read_csv(f'{folder}/{fname}', skiprows=28)[:-1]
        cols = [
            'angle_sin', 'angle_cos', 'angleD', 'position', 
            'positionD', 'target_equilibrium', 'target_position'
        ]
        if include_L:
            cols += ['L']
        x = df[cols]
        y = df[['Q']]
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def load_train_test_data_27s(include_L=False):
    folder = 'DG-27s-and-1s500ms-noisy-u/Recordings/Train/Train-27s'
    x_train, y_train = load_data(folder, include_L=include_L)
    folder = 'DG-27s-and-1s500ms-noisy-u/Recordings/Test/Test-27s'
    x_test, y_test = load_data(folder, include_L=include_L)
    return x_train, y_train, x_test, y_test

def load_train_test_data(include_L=False):
    folder = 'DG-27s-and-1s500ms-noisy-u/Recordings/Train/Train-1s500ms'
    x_train, y_train = load_data(folder, include_L=include_L)
    folder = 'DG-27s-and-1s500ms-noisy-u/Recordings/Test/Test-1s500ms'
    x_test, y_test = load_data(folder, include_L=include_L)
    return x_train, y_train, x_test, y_test

# ------------------------------------------------------------------------------

def train_model(hidden_size, memory_size, theta, epochs=500, data='500ms'):
    if data == '500ms':
        print('using 500ms data')
        x_train, y_train, x_test, y_test = load_train_test_data()
    else:
        print('using 27s data')
        x_train, y_train, x_test, y_test = load_train_test_data_27s()

    model = LMUModel(input_size=7, output_size=1, hidden_size=hidden_size, 
                     memory_size=memory_size, theta=theta)
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()

    batch_size = 64
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs+1):
        # eval
        if epoch % 50 == 0:
            loss_train = []
            for batch_idx in range(n_batches-1):
                model.eval()
                x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_train.append(loss(ypr, y).item())
            loss_test = []
            for batch_idx in range(max(1, x_test.shape[0] // batch_size)):
                model.eval()
                x = x_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_test.append(loss(ypr, y).item())
            print(epoch, 'train', np.array(loss_train).mean(), 'test', np.array(loss_test).mean())

        # train
        epoch_loss = []
        model.train()
        for batch_idx in range(n_batches-1):
            x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

            epoch_loss.append(l.item())

        # log training loss
        avg_epoch_loss = np.array(epoch_loss).mean()
        if epoch % 10 == 0:
            print(epoch, avg_epoch_loss)
        else:
            print(epoch, avg_epoch_loss, end='\r')

    model_name = f'mpc_models/lmu_7-1-{hidden_size}-{memory_size}-{theta}.pt'
    torch.save(model.state_dict(), model_name)

def load_model(model_name):
    hidden_size = int(model_name.split('-')[2])
    memory_size = int(model_name.split('-')[3])
    theta = float(model_name.split('-')[4].split('.')[0])
    model = LMUModel(input_size=7, output_size=1, hidden_size=hidden_size,
                     memory_size=memory_size, theta=theta)
    model.load_state_dict(torch.load(model_name))
    return model

def train_mlp(n_neurons=64, epochs=500):
    x_train, y_train, x_test, y_test = load_train_test_data()
    x_train = x_train.reshape(-1, 7)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 7)
    y_test = y_test.reshape(-1, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(7, n_neurons),
        torch.nn.ReLU(),
        torch.nn.Linear(n_neurons, 1)
    )
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()

    batch_size = 512
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs+1):
        # eval
        if epoch % 50 == 0:
            loss_train = []
            for batch_idx in range(n_batches-1):
                model.eval()
                x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_train.append(loss(ypr, y).item())
            loss_test = []
            for batch_idx in range(x_test.shape[0] // batch_size):
                model.eval()
                x = x_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y = y_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                ypr = model(x)
                loss_test.append(loss(ypr, y).item())
            print(epoch, 'train', np.array(loss_train).mean(), 'test', np.array(loss_test).mean())

        # train
        epoch_loss = []
        model.train()
        for batch_idx in range(n_batches-1):
            x = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

            epoch_loss.append(l.item())

        # log training loss
        avg_epoch_loss = np.array(epoch_loss).mean()
        if epoch % 10 == 0:
            print(epoch, avg_epoch_loss)
        else:
            print(epoch, avg_epoch_loss, end='\r')

    model_name = f'mpc_models/mlp_7-1-{n_neurons}.pt'
    torch.save(model.state_dict(), model_name)
