from utils import torch,nn,F,optim,datasets,transforms
import utils

class Net(nn.Module):   
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()    
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(1, 4, 3), #input -28 OUtput-26 RF-3
            nn.ReLU(), 
           
            nn.Conv2d(4, 4, 3), #input -26 OUtput-24 RF-5
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), #input -24 OUtput-12 RF-6
            
            nn.Conv2d(4, 8, 3), #input -12 OUtput-10 RF-10
            nn.ReLU(),
            
            nn.Conv2d(8, 8, 3), #input -10 OUtput-8 RF-14
            nn.ReLU(),
            
            nn.Conv2d(8, 16, 3), #input -8 OUtput-6 RF-18
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 3), #input -6 OUtput-4 RF-22
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3), #input -4 OUtput-2 RF-26
            nn.ReLU(),           
            
                      
            nn.Conv2d(64, 10, 1), #input -2 OUtput-2 RF-26, antman-squeeze
            nn.AvgPool2d(2),
               
           
        )
        
       
       
    def forward(self, x):       
        x = self.conv1(x)      
        x = x.view(-1, 10)
        return F.log_softmax(x)      
    
def getModel():
    return Net().to(utils.getDevice())


    

    
    
     