import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
from Perceptron.data_loader import load_X_Y,load_data,norm_data
from flask_socketio import SocketIO, emit

class SakuraPerceptron(nn.Module):
    def __init__(self):
        super(SakuraPerceptron,self).__init__()
        self.layer1 = nn.Linear(43,64)
        self.layer2 = nn.Linear(64,1)
        
    def forward(self,x):
        x = torch.sigmoid(self.layer1(x.float()))
        x = self.layer2(x.float())
        return x
    
def train(socketio, train_data_path,train_data_label_path,batch_size,learning_rate = 0.01):
    # load data for training    
    data_raw,label = load_data(train_data_path,train_data_label_path)
    data_raw = norm_data(data_raw,"avg_temp")
    
    # label = norm_data(label,"day")
    X, Y = load_X_Y(data_raw,label)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(f"X: {X.shape}, Y: {Y.shape}")


    # build loader for training
    loader = DataLoader(list(zip(X,Y)),batch_size=batch_size,shuffle=True)

    # setup model
    sakura_model = SakuraPerceptron()
    criterion = nn.MSELoss() # loss function
    optimizer = optim.SGD(sakura_model.parameters(),lr=learning_rate)

    # start training
    num_epochs = 2000
    # num_epochs = 200
    loss_record = [] # the array for recording loss each batch
    special_loss_record = [] # the array for recording loss each epoch

    for epoch in range(num_epochs):
        # print(f"Epoch:",epoch)

        sakura_model.train() # set model to train mode
        running_loss = 0.0

        for i,(x_batch, y_batch) in enumerate(loader):

            optimizer.zero_grad()
            output = sakura_model(x_batch).float()
            loss = criterion(output,y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # save running loss

            # print current loss every 30 batches
            if i % (34/batch_size) == (34/batch_size)-1:
                # print('[%d, %5d] loss: %.3f' %
                #   (epoch + 1, i + 1, running_loss / (34/batch_size)))

                loss_record.append(running_loss/(34/batch_size))

        # print current loss if current epoch num in the list

        special_epoch = [*range(0,2000,10)]
        if epoch + 1 in special_epoch:
            print("--------------------------------")
            print(running_loss/len(loader))
            socketio.emit('training_loss', {'Epoch':epoch, 'loss':running_loss/len(loader)})
            socketio.sleep(0.1)
            

    # save loss 
    # now = str(datetime.now()).replace(" ","")
    # pd.DataFrame({"loss":loss_record}).to_csv(f"loss/loss_{now}.csv",index=True)
    # pd.DataFrame({"epoch":special_epoch,"loss":special_loss_record}).to_csv(f"loss/special_loss_record_{now}.csv",index=True)

    print('Finished Training')    

    # save model
    torch.save(sakura_model.state_dict(),"sakura_model_second.pt")