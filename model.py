import torch
import torch.nn as nn


''' Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)'''
class Generator(nn.Module):
    def __init__(self,nosie_dim = 50):
        super(Generator,self).__init__()        
        self.g_fc1 = nn.Sequential(
                    nn.Linear(nosie_dim,1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU()
        )
        self.g_fc2 = nn.Sequential(
                    nn.Linear(1024,128*7*7),
                    nn.BatchNorm1d(128*7*7),
                    nn.LeakyReLU()
        )
        self.g_dc3 = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels= 128,     
                        out_channels= 64,    
                        kernel_size=2,     
                        stride=2,           
                        padding=0,      
                    ),                    
                    nn.LeakyReLU(),
                )      
        self.g_dc4 = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels= 64,     
                        out_channels= 1,    
                        kernel_size=2,     
                        stride=2,           
                        padding=0,      
                    ),                    
                    nn.Sigmoid(),
                )  
    def forward(self,input_tensor):       
        x = self.g_fc1(input_tensor)   
        x = self.g_fc2(x)
        x = x.view(-1,128,7,7)
        x = self.g_dc3(x)
        x= self.g_dc4(x)
        return x
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(
                        in_channels = 1,     
                        out_channels = 64,    
                        kernel_size = 4,      
                        stride = 2,           
                        padding = 1,      
                    ),      
                    nn.LeakyReLU(),   
                    nn.Conv2d(
                        in_channels = 64,     
                        out_channels = 128,    
                        kernel_size = 4,      
                        stride = 2,           
                        padding = 1,      
                    ),  
                    nn.BatchNorm2d(128),        
                    nn.LeakyReLU(),    
                )
        self.fc2 = nn.Sequential(  
                nn.Linear(128*7*7,1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),    
                nn.Linear(1024,1),
                nn.Sigmoid()
                )
    def forward(self,input_tensor):        
        x = self.conv1(input_tensor)
        x = x.view(input_tensor.size()[0],-1)
        x = self.fc2(x)
        return x
    
        
