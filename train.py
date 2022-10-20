import torch
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,1,0)
    score = torch.where(y_pred==y_true,1.0,0.0)
    score = torch.mean(score)
    return score  


import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
def train_model(data_loader, model, optimizer,epoch,w):
    num_batches = len(data_loader)
    model.train()
    total_bce = 0
    total_accu = 0
    
    for x, y in data_loader:
        #x = x.reshape([x.shape[0]*x.shape[1],1,x.shape[-1]]).squeeze(dim = 1)
        #y = y.reshape([y.shape[0]*y.shape[1],1,y.shape[-1]]).squeeze(dim = 1)
        X,Y = x.to(device).float(),y.to(device).float()
        X.requires_grad=True
        Y.requires_grad=True
        output = model(X)

        bce = F.binary_cross_entropy(output, Y)
        accu = accuracy(output,Y)
        
        #a=X.grad

        optimizer.zero_grad()
        bce.backward()
        #torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=2, norm_type=2.0)
        optimizer.step()
        

        total_bce += bce.item()    
        total_accu += accu
    #print(a)
    #print(b)
    
    avg_bce = total_bce / num_batches
    avg_accu = total_accu / num_batches
    
    w.add_scalar('Training_BCE', avg_bce, epoch)
    w.add_scalar('Training_Accuracy', avg_accu, epoch)

    print(f"Train BCE: {avg_bce}      Train Accuracy: {avg_accu}")




def validation_model(data_loader, model,epoch,w):

    num_batches = len(data_loader)
    total_bce = 0
    total_accu = 0
    accu = torch.Tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            X,Y = x.to(device).float(),y.to(device).float()
            output = model(X)

            total_bce += F.binary_cross_entropy(output, Y).item()
            total_accu = accuracy(output, Y)
            
            

    avg_bce = total_bce / num_batches
    avg_accu = total_accu 
    #/ num_batches

    w.add_scalar('Validation_BCE', avg_bce, epoch)
    w.add_scalar('Validation_Accuracy', avg_accu, epoch)
    
    

    print(f"Validation BCE: {avg_bce}   Validation Accuracy: {avg_accu}")



def fit(model,train_loader,valid_loader,epochs,optimizer,name,learning_rate):
    #epochs = 50
    
    w = SummaryWriter(name,comment=f'{learning_rate}')
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_model(train_loader, model, optimizer,t,w)
        validation_model(valid_loader, model, t,w)

    X, y = next(iter(train_loader))
    X = X.to(device).float()
    w.add_graph(model, input_to_model=X, verbose=False, use_strict_trace=True)
    w.close()
    #pred = make_prediction(test_loader, model,w)
    print("Done!")
    return model