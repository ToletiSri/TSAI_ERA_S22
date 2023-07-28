from utils import torch,nn,F,optim,datasets,transforms
import utils

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Preparation Layer
        self.convblockPreparation = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 32x32x64, RF = 3
        
        
        # Layer 1
        # Layer1, X = 
        self.convblockL1X1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 5
            nn.MaxPool2d(2, 2),            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 16x16x128; RF = 6
        
        #Layer1, res1 = 
        self.convblockL1R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 10           
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # output_size = 16x16x128; RF = 10
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),           
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )  # output_size = 16x16x128; RF = 14
            
        
        # Layer 2       
        self.convblockL2X1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 18
            nn.MaxPool2d(2, 2), #RF = 20            
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 8x8x256; RF = 20
            
        
        # Layer 3
        # Layer3, X = 
        self.convblockL3X1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 28
            nn.MaxPool2d(2, 2),  #RF = 32           
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        ) # output_size = 4x4x512; RF = 32
        #Layer3, res1 = 
        self.convblockL3R1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),  #RF = 50          
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # output_size = 4x4x512; RF = 50
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),   #RF = 66            
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )  # output_size = 4x4x512; RF = 66
            

        #LayerFinal 
        self.FinalBlock = nn.Sequential(
             nn.MaxPool2d(4, 4),  ## output_size = 1x1x512; RF = 74             
        )
        
        #LayerFC
        self.FC = nn.Sequential(
             nn.Linear(512, 10),  ## output_size = 1x1x512; RF = 74             
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        #print('shape of x before preparation = ', x.shape)
        #Prep
        x = self.convblockPreparation(x)
        
        #print('Shape after preparation = ',x.shape)
        #L1
        x = self.convblockL1X1(x)
        #print('Shape after L1 X1 = ',x.shape)
        x = x + self.convblockL1R1(x)
        #print('Shape after L1 R1 = ',x.shape)
        
        #L2
        x = self.convblockL2X1(x)
        #print('Shape after L2 = ',x.shape)

        #L3
        x = self.convblockL3X1(x)
        #print('Shape after L3 X1 = ',x.shape)
        x = x + self.convblockL3R1(x)
        #print('Shape after L3 R1 = ',x.shape)
      
        #Final       
        x = self.FinalBlock(x)
        #print('Shape after final = ',x.shape)

        x = x.view(-1, 512)
        #print('Shape after view = ',x.shape)
        
        x = self.FC(x)
        #print('Shape after FC = ',x.shape)
        return x.view(-1, 10)
        
        #return nn.Linear(512, 10)
        #print('Shape after Linear = ',x.shape)

        #return F.log_softmax(x, dim=-1)
    
    
def getModel():
    return Net().to(utils.getDevice())


    

    
    
     