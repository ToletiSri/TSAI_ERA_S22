from utils import torch,nn,F,optim,datasets,transforms
import utils

class Net(nn.Module):   
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()    
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(1, 8, 3, bias = False), #input -28 OUtput-26 RF-3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.02),
           
            nn.Conv2d(8, 16, 3, bias = False), #input -26 OUtput-24 RF-5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.02),
            
            #Transition layer
            nn.MaxPool2d(2, 2), #input -24 OUtput-12 RF-6
            nn.Conv2d(16, 8, 1), #input -12 OUtput-12 RF-6, antman-squeeze
            nn.BatchNorm2d(8),
           
            
            nn.Conv2d(8, 16, 3, bias = False), #input -12 OUtput-10 RF-10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.02),
            
            nn.Conv2d(16, 20, 3, bias = False), #input -10 OUtput-8 RF-14
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.02),    
            
             #Transition layer
            nn.MaxPool2d(2, 2), #input -8 OUtput-4 RF-16
            nn.Conv2d(20, 16, 1), #input -4 OUtput-4 RF-16, antman-squeeze
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 10, 3, bias = False), #input -4 OUtput-2 RF-24
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.02),  
           
            
            nn.AvgPool2d(2), #input -2 OUtput-1 RF-28                                                         
    
               
           
        )
        
       
       
    def forward(self, x):       
        x = self.conv1(x)   
        x = x.view(-1, 10)
        return F.log_softmax(x)      
    
def getModel():
    return Net().to(utils.getDevice())


    

    
    
     