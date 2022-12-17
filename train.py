import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from load_data import get_data

# Create ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        # Linear function 1: len(fetures_name)+2 --> 16
        self.fc1 = nn.Linear(input_dim, hidden_dim[0]) 
        # Non-linearity 1
        self.relu1 = nn.ReLU(inplace=False)
        
        # Linear function 2: 16 --> 32
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        # Non-linearity 2
        self.tanh2 = nn.Tanh()
        
        # Linear function 3 (readout): 32 --> 1
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)
        
        # Linear function 3 (readout)
        out = self.fc3(out)
        
        # compress to one-dimension
        out = out.squeeze(-1)

        return out

class MSE3Loss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(MSE3Loss,self).__init__()
        
    def forward(self, epspre, gmpre, gdpre, labels, a1, a2):
    #def forward(self, epspre,labels):
        epspre0 = epspre.clone()
        gmpre0 = gmpre.clone()
        gdpre0 = gdpre.clone()
        epstar = labels[:,0].clone()
        gmtar = labels[:,1].clone()
        gdtar = labels[:,2].clone()
        gmvstar = labels[:,3].clone()
        gdvstar = labels[:,4].clone()
        idvstar = labels[:,5].clone()

        loss = 1/3 * torch.mean((epspre0 - epstar)**2 + a1 * torch.mean(((gmpre0*idvstar + gmvstar*epspre0) - gmtar)**2) + a2 * torch.mean(((gdpre0*idvstar + gdvstar*epspre0) - gdtar)**2))
        #loss = torch.mean((epspre0 - epstar)**2)
        return loss

def pre_data(dir, train_or_test, device):
    para_numpy, eps_numpy, gm_numpy, gd_numpy,  gm_vs_numpy, gd_vs_numpy, id_vs_numpy, vdd, vlin, features_name, lowerbound, upperbound = get_data(dir, train_or_test)
    # 需要注意的是features的最后两个参数是vd，不作为神经网络的输入，但是还是具有作用
    # 转化为float32
    features_numpy = para_numpy[:,0:len(features_name)+2].astype(np.float32) 
    print(len(features_numpy))

    # vgvd_numpy = para_numpy[:,len(features_name)+2:-1].astype(np.float32) 
    # index_file = para_numpy[:,-1] #TCAD数据的标签
    
    eps_numpy = eps_numpy.astype(np.float32)
    gm_numpy = gm_numpy.astype(np.float32)
    gd_numpy = gd_numpy.astype(np.float32)
    gm_vs_numpy = gm_vs_numpy.astype(np.float32)
    gd_vs_numpy = gd_vs_numpy.astype(np.float32)
    id_vs_numpy = id_vs_numpy.astype(np.float32)

    eps_numpy = eps_numpy.reshape(-1,1) #转变位列向量
    gm_numpy = gm_numpy.reshape(-1,1)
    gd_numpy = gd_numpy.reshape(-1,1)
    gm_vs_numpy = gm_vs_numpy.reshape(-1,1)
    gd_vs_numpy = gd_vs_numpy.reshape(-1,1)
    id_vs_numpy = id_vs_numpy.reshape(-1,1)

    print(eps_numpy.size)
    print(gm_numpy.size)
    print(gd_numpy.size)
    print(gm_vs_numpy.size)
    print(gd_vs_numpy.size)
    print(id_vs_numpy.size)

    targets_list = []  #为了确保跨导电导的数据在之后可以被找到，需要和eps一同放入label中
    for i in range(len(eps_numpy)):
        inter =[eps_numpy[i][0],gm_numpy[i][0],gd_numpy[i][0],gm_vs_numpy[i][0],gd_vs_numpy[i][0],id_vs_numpy[i][0]]
        targets_list.append(inter)
    targets_numpy = np.array(targets_list)
    print(len(targets_numpy))

    # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
    features = torch.from_numpy(features_numpy).to(device)
    targets = torch.from_numpy(targets_numpy).to(device)
    # print('features.shape:',features.shape,'targets.shape:',targets.shape)
    return features, targets, vdd, vlin, features_name, lowerbound, upperbound

def get_derivate(feature, dv):
    '''
        feature is a tensor, feature[:,-1] is u1, feature[:,-2] is u2
        gm is tran-con, relate to vgs
        gd is conduct-, relate to vds
        u1 = 2 * vgs - vds
        u2 = vds^2
    '''
    gm = feature.clone()
    gd = feature.clone()
    u1_gm = gm[:,-2] + 2*dv
    gm[:,-2] = u1_gm

    u1_gd = gd[:,-2] - dv
    u2_gd = torch.pow((torch.sqrt(gd[:,-1]) + dv), 2)
    gd[:,-2] = u1_gd
    gd[:,-1] = u2_gd
    return gm, gd

def train_main(dir0):
    #GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir = dir0

    features_train, targets_train, vdd_train, vlin_train, features_name_train, lowerbound_train, upperbound_train = pre_data(dir, 0, device)
    features_test, targets_test, vdd_test, vlin_test, features_name_test, lowerbound_test, upperbound_test = pre_data(dir, 1, device)
    # batch_size, epoch and iteration
    batch_size = 100
    #n_iters = 10000
    #num_epochs = n_iters / (len(features_train) / batch_size)
    num_epochs = 20
    
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(features_train,targets_train)
    test = torch.utils.data.TensorDataset(features_test,targets_test)
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    # instantiate ANN
    input_dim = len(features_name_train) + 2
    hidden_dim = [16,32] #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
    output_dim = 1

    # Create ANN
    model = ANNModel(input_dim, hidden_dim, output_dim).to(device)

    # Cross Entropy Loss 
    error = MSE3Loss().to(device)

    # SGD Optimizer
    learning_rate = 0.02
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # ANN model training
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
  
    for epoch in range(num_epochs):
        for i, (train, train_target) in enumerate(train_loader):
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(train)

            # get derivate
            gm_input, gd_input = get_derivate(train, 0.05) #由于 u1 = f1(vgs,vds),u2 = f2(vgs,vds)作为网络输入，所以训练时需要先进行转化，求出在原本的u1,u2上变化了Vds，Vgs之后的u1,u2
            outputs_gm = model(gm_input)
            outputs_gd = model(gd_input)
            gmpre = (outputs_gm - outputs) / 0.05 
            gdpre = (outputs_gd - outputs) / 0.05

            # Calculate softmax and ross entropy loss
            a = 1 #分布描述跨导、电导在损失函数中的重要性
            b = 1
            loss = error(outputs,gmpre,gdpre,train_target,a,b)
            #loss = error(outputs,train_target)
            # Calculating gradients
            #loss.backward(retain_graph=True)
            loss.backward()

            # Update parameters
            optimizer.step()
            
            count = count + 1
            
            if count % 50 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Predict test dataset
                test_output0 = 0
                test_target0 = 0
                for test, test_target in test_loader:
                    
                    # Forward propagation
                    outputs = model(test)
                    
                    total = total + len(outputs)
                    # for ii in range(1):
                    #     print('test_output:', outputs[ii], ' test_target:',test_target[ii,0])

                    comp = torch.abs((test_target[:,0] - outputs)/ test_target[:,0])
                    # Total correct predictions
                    correct = correct + comp.sum()

                    test_output0 = outputs[0]
                    test_target0 = test_target[0,0]
                
                accuracy = correct / int(total)
                
                # store loss and iteration
                loss_list.append(loss.cpu().data.numpy())
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                if count % 500 == 0:
                    # Print Loss

                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
                    print('test_output:',test_output0,' test_target:',test_target0)

if __name__ == '__main__':   
    dir = r'C:\Users\xxxin\Documents\GitHub\DataBase\EDA2022\GAA_data'
    train_main(dir)
    
    # loss_list = np.array(loss_list)
    # iteration_list = np.array(iteration_list)
    # np.savetxt('loss_list_train2.txt',loss_list)
    # np.savetxt('iteration_list_train2.txt',iteration_list)

    ''' 
    targets_list = []
    predict_list = []
    test_loss_list = []
    with torch.no_grad():
        for test, labels in test_loader:
            # Forward propagation
            outputs = model(test)
            targets_list.append(labels.cpu().data.numpy())
            predict_list.append(outputs.cpu().data.numpy())
            loss = error(outputs, labels)
            test_loss_list.append(loss.cpu().data.numpy())
    targets_list.reshape(-1)
    predict_list.reshape(-1)
    test_loss_list.reshape(-1)
    df_test = pd.DataFrame({'targets':targets_list,'predict':predict_list,'loss':test_loss_list})
    df_test.to_csv('test_output.csv',index = False)
    '''