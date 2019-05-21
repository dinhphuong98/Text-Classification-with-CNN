
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

import pickle
##-------------------------------------##

class Net(nn.Module):
    
    def __init__(self,num_class):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.mp = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1472, 640)
        torch.nn.init.xavier_uniform(self.fc.weight)
        self.fc1 = nn.Linear(640, num_class)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc1(x)
       
        return F.log_softmax(x)

def convert_out_to_class(out):
    '''
        out is a pytorch tensor
    '''
    out = out.detach()[0].tolist()
    out_max = max(out)
    return out.index(out_max)

class MyDataset(Dataset):
    def __init__(self,root_path = 'dataset.pkl'):
        with open(root_path, 'rb') as f:
            mynewlist = pickle.load(f)
        self.X_train = []
        self.Y_train = []
        for data in mynewlist:
            x,y = data
            self.X_train.append(x)
            self.Y_train.append(y)
       
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        self.len = len(self.X_train)
        return self.len


  
dataset = MyDataset('train_dataset (1).pkl')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True)

test_dataset = MyDataset('test_dataset.pkl')
test_loader =  DataLoader(dataset=test_dataset,
                          batch_size=32,
                          shuffle=True)

temp = torch.rand([32,1,15,100])

def test(model,loss,e,test_loader = test_loader):
  
    model.eval()
    test_loss = 0
    correct = 0
    for data in test_loader:
        x, y = data
        y = y.view(-1)
            
        if x.size() != temp.size():
            continue
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        
        test_loss += loss(output, y).item()
      
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    if e % 10 == 0:
      print('\nTest set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    
    num_class = 10
    epochs = 20
   
    model = Net(num_class = num_class)
    model = model.cuda()
   
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters())
    print('Training...')
    print('Data train of sample: {}'.format(len(train_loader)))
    print('Data test of sample: {}'.format(len(test_loader)))
    train_losses = []
    train_acc = []
    vali_losses = []
    vali_acc = []
   
    for e in range(epochs):
        
        sum_loss = 0
        correct = 0
        for data in train_loader:
            model.train()
            opt.zero_grad()
            
            x,y = data
            y = y.view(-1)
            
            if x.size() != temp.size():
                continue
            x = x.cuda()
            y = y.cuda()
           
            
            out = model(x)
            l = loss(out,y)
            
            
            pred = out.detach().data.max(1, keepdim=True)[1]
            correct += pred.eq(y.detach().data.view_as(pred)).cpu().sum()
            
            sum_loss+=l.item()
            l.backward()
            opt.step()
            
        
        train_losses.append(sum_loss)
        train_acc.append(100. * correct / (len(train_loader.dataset)))
        if e % 10 == 0:
          print('Epoch: {} - loss {} - Acc {}%'.format(e,sum_loss,100. * correct / (len(train_loader.dataset))))
        t_l,t_a = test(model,loss,e)
        vali_losses.append(t_l)
        vali_acc.append(t_a)
   
    plt.plot(train_losses)
    plt.title('Tranin Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(train_acc)
    plt.title('Train Data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    plt.plot(vali_losses)
    plt.title('Test Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(vali_acc)
    plt.title('Test Data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    filepath = 'pretrained'

    torch.save(model.state_dict(),filepath)
    
    
