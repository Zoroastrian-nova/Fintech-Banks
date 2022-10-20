import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
class Logistic(torch.nn.Module):
    def __init__(self,n) -> None:
        super(Logistic,self).__init__()
        self.regression = torch.nn.Linear(n,out_features=1,bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x= self.regression(x)
        output= self.sigmoid(x)
        return output

class MLP(torch.nn.Module):
    def __init__(self,n,num_layer) -> None:
        super(MLP,self).__init__()
        
        
        self.relu = torch.nn.ReLU()
        self.dense = torch.nn.Sequential()

        for i in range(num_layer):
            if i==0:
                self.dense.append(torch.nn.Linear(in_features=n,out_features=512))
                self.dense.append(self.relu)
            else:
                self.dense.append(torch.nn.Linear(in_features=256*2**i,out_features=256*2**(i+1)))
                self.dense.append(self.relu)
        self.regress = torch.nn.Linear(in_features= 256*2**(num_layer),out_features= 1 )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x= self.dense(x)
        x= self.regress(x)
        output= self.sigmoid(x)
        return output

from torch.autograd import Variable
class LSTM(torch.nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.normalization = torch.nn.BatchNorm1d(num_features= num_classes)

        self.lstm_1 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=False) #lstm
        self.fc_1 =  torch.nn.Linear(hidden_size, hidden_size) #fully connected 1

        self.fc = torch.nn.Linear(hidden_size, num_classes) #fully connected last layer

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers,x.size(1),  self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)).to(device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm_1(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #hn = hn[-1, :,:].reshape(1,x.size(1), self.hidden_size).squeeze(dim = 0) #reshaping the data for Dense layer next
        #cn = cn[-1, :,:].reshape(1,x.size(1), self.hidden_size).squeeze(dim = 0) #reshaping the data for Dense layer next
        out = self.relu(output)
        out = self.fc_1(out) #first Dense

        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        out = self.sigmoid(out) #relu
        return out

class tfm(torch.nn.Module):
     def __init__(self,n,encoder_layer):
         super().__init__()
         self.n_encoder = encoder_layer
         self.encoder = torch.nn.Sequential()
         
         self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model = n, nhead = 2, 
         dim_feedforward=64, dropout=0.1, activation='relu', 
         layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
         self.relu1 = torch.nn.ReLU()
         for _ in range(self.n_encoder):
            self.encoder.append(self.encoder_layer)
            self.encoder.append(self.relu1)

         self.fc2 = torch.nn.Linear(in_features=n,out_features=1)
         self.sigmoid = torch.nn.Sigmoid()
 
     def forward(self, x):

        x = self.relu1(self.encoder(x))
        y = self.sigmoid(self.fc2(x))
        return y

from torch import nn
#from torch import functional as F
num_channels = 13
kernel_size_1 = 13
groups = 1
depth_1 = 13
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels= num_channels,out_channels= depth_1, kernel_size=kernel_size_1, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=kernel_size_1, stride=1),
                nn.Dropout(0.1),
        )
        #print(self.conv1)


        self.linear = nn.Sequential(
            nn.Linear(208, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x= self.conv1(x)
        #x = self.flatten(x)
        #x = x.view(x.size(0), -1)
        #x = torch.reshape(input= = x,shape=[])
        logits = self.linear(x)
        return logits