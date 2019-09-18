import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision     


torch.manual_seed(1)    
class load_data:
    def __init__(self,epoch=1,batch_size=50):
        self.EPOCH = epoch           
        self.BATCH_SIZE = batch_size
    def train(self):        
        train_data = torchvision.datasets.MNIST(
            root='./MNIST_data/',    
            train=True,  
            transform=torchvision.transforms.ToTensor(),
            download=True,                                                      
        )        
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)
        return train_loader
    def test(self):
        test_data = torchvision.datasets.MNIST(root='./MNIST_data/', train=False,download=True)        
        test_VAL = test_data.test_data[0:5000].view(-1,1,28,28).type(torch.FloatTensor)/255.   
        test_TEST =test_data.test_data[5000:10000].view(-1,1,28,28).type(torch.FloatTensor)/255.       
        return test_VAL,test_TEST


    