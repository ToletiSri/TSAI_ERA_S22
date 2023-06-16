from utils import torch,nn,F,optim,datasets,transforms
import utils

class Net(nn.Module):   
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()    
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(1, 2, 3), #input -28 OUtput-26 RF-3
            nn.ReLU(), 
           
            nn.Conv2d(2, 4, 3), #input -26 OUtput-24 RF-5
            nn.ReLU(),
            
            nn.Conv2d(4, 8, 3), #input -24 OUtput-22 RF-7
            nn.ReLU(),
            
            nn.Conv2d(8, 16, 3), #input -22 OUtput-20 RF-9
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 3), #input -20 OUtput-18 RF-11
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3), #input -18 OUtput-16 RF-13
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3), #input -16 OUtput-14 RF-15
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3), #input -14 OUtput-12 RF-17
            nn.ReLU(),
            
            nn.Conv2d(256, 512, 3), #input -12 OUtput-10 RF-19
            nn.ReLU(),
            
            nn.Conv2d(512, 1024, 3), #input -10 OUtput-8 RF-21
                      
            nn.Conv2d(1024, 10, 1), #input -8 OUtput-8 RF-21, antman-squeeze
            nn.AvgPool2d(8),
               
           
        )
        
       
       
    def forward(self, x):       
        x = self.conv1(x)      
        x = x.view(-1, 10)
        return F.log_softmax(x)      
    
def getModel():
    return Net().to(utils.getDevice())


    

    
    
     