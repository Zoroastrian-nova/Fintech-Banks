import pandas as pd
import numpy as np
import torch 

import utils
import models
import train

from torch.utils.data import DataLoader
from torchinfo import summary
def main():
    SEED = 3407
    utils.seed_everything(SEED)
    size = 100000
    dataset = utils.load_data(size)
    dataset = utils.preprocess(dataset)

    df_train,df_valid = dataset[:int(2/3*len(dataset))],dataset[int(2/3*len(dataset)):]
    fea = [c for c in dataset.columns if c not in ['customer_ID','S_2','target_x','target_y','target_x']]
    lab = ['target_x']
    n = len(fea)

    batch_size= 256

    input_length = 13
    output_length = 13
    train_dataset = utils.SequenceDataset(
    df_train,
    target=lab,
    features=fea,
    x_length=input_length,
    y_length=output_length,
    )

    valid_dataset = utils.SequenceDataset(
    df_valid,
    target=lab,
    features=fea,
    x_length=input_length,
    y_length=output_length,
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logistic_model = models.Logistic(n).float().to(device)
    lr = 1e-3
    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=lr,momentum=0.9,weight_decay=0.005)
    summary(logistic_model)
    #logistic_model = train.fit(logistic_model,train_loader,valid_loader,epochs=100,optimizer=optimizer,name=f'./graphs/Logistic{lr}',learning_rate=lr)
    #torch.save(logistic_model.state_dict(), './model/logistic_model.pth')

    mlp_model = models.MLP(n,2).float().to(device)
    lr = 1e-3
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=lr,momentum=0.9,weight_decay=0.005)
    summary(mlp_model)
    mlp_model = train.fit(mlp_model,train_loader,valid_loader,epochs=100,optimizer=optimizer,name=f'./graphs/MLP{lr}',learning_rate=lr)
    torch.save(mlp_model.state_dict(), './model/mlp_model.pth')

    input_size = len(fea) #number of features
    hidden_size = len(fea) #number of features in hidden state
    num_classes = 1 #number of output classes
    seq_length = 13
    num_layers = 1 #number of stacked lstm layers
    lstm_model = models.LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
    lstm_model = lstm_model.float().to(device)
    lr = 1e-2
    optimizer = torch.optim.SGD(lstm_model.parameters(), lr=lr,momentum=0.9,weight_decay=0.005)
    summary(lstm_model)
    lstm_model = train.fit(lstm_model,train_loader,valid_loader,epochs=50,optimizer=optimizer,name=f'./graphs/LSTM{lr}',learning_rate=lr)
    torch.save(lstm_model.state_dict(), './model/lstm_model.pth')

    tfm_model = models.tfm(n=n,encoder_layer = 2).to(device)
    lr = 1e-3
    optimizer = torch.optim.SGD(tfm_model.parameters(), lr=lr,momentum=0.9,weight_decay=0.005)
    summary(tfm_model)
    tfm_model = train.fit(tfm_model,train_loader,valid_loader,epochs=100,optimizer=optimizer,name=f'./graphs/Transformer{lr}',learning_rate=lr)
    torch.save(tfm_model.state_dict(), './model/tfm_model.pth')

    cnn_model = models.CNN().to(device)
    cnn_model = cnn_model.float()
    lr = 1e-3
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr,weight_decay = 0.1)
    summary(cnn_model)
    cnn_model = train.fit(cnn_model,train_loader,valid_loader,epochs=100,optimizer=optimizer,name=f'./graphs/CNN{lr}',learning_rate=lr)
    torch.save(cnn_model.state_dict(), './model/cnn_model.pth')


if __name__ == "__main__":
    main()