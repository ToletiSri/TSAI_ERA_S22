from utils import torch,nn,F,optim,datasets,transforms
import utils

class Net(nn.Module):   
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()    
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), #input -28 OUtput-28 RF-3
            nn.ReLU(), 
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1), #input -28 OUtput-28 RF-5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01),
            
            nn.MaxPool2d(2, 2), #input -28 OUtput-14 RF-6
            nn.Conv2d(16, 8, 1), #input -14 OUtput-14 RF-6, antman-squeeze
            #nn.ReLU(),
            nn.BatchNorm2d(8),
                        
            nn.Conv2d(8, 16, 3, padding=1), #input -14 OUtput-14 RF-10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1), #input -14 OUtput-14 RF-14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01),
            
            nn.MaxPool2d(2, 2), #input -14 OUtput-7 RF-16
            nn.Conv2d(32, 16, 1), #input -7 OUtput-7 RF-16, antman-squeeze
            #nn.ReLU(),
            nn.BatchNorm2d(16),
           
           
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  #input - 7 OUtput-7 RF-24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Conv2d(32, 16, 3, padding=1), #input -7 OUtput-7 RF-32
            nn.BatchNorm2d(16),            
            nn.Conv2d(16, 10, 1), #input -7 OUtput-7 RF-32, antman
            nn.BatchNorm2d(10),
            nn.AvgPool2d(7),
           
        )
        
        #self.conv1 = nn.Conv2d(1, 4, 3, padding=1) #input -28 OUtput-28 RF-3
        #self.conv2 = nn.Conv2d(4, 8, 3, padding=1) #input -28 OUtput-28 RF-5 
        #self.pool1 = nn.MaxPool2d(2, 2)  #input -28 OUtput-14 RF-6                     
        #self.conv3 = nn.Conv2d(8, 16, 3, padding=1) #input -14 OUtput-14 RF-10
        #self.conv4 = nn.Conv2d(16, 32, 3, padding=1) #input -14 OUtput-14 RF-14                    
        #self.pool2 = nn.MaxPool2d(2, 2)  #input -14 OUtput-7 RF-15
        #self.conv5 = nn.Conv2d(32, 64, 3,padding=1) #input -7 OUtput-7 RF-23
        #self.conv6 = nn.Conv2d(64, 128, 3, padding=1) #input -7 OUtput-7 RF-31; 7*7*512
        #self.conv7 = nn.Conv2d(128, 10,1,padding=1)  #input -7 OUtput-7 RF-31; 
        #self.pool3 = nn.AvgPool2d(7)  #input -7 OUtput-1 RF-31
       
    def forward(self, x):
        #x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        #x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        #x = F.relu(self.conv6(F.relu(self.conv5(x))))
        #x = self.conv7(x)
        #x = self.pool3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)      
    
def getModel():
    return Net().to(utils.getDevice())


    

    
    
     