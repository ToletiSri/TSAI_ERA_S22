from utils import torch,nn,F,optim,datasets,transforms
import utils

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1 input 32/1/1
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/3
        
        self.convblock112 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/3
        
        self.convblock113 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/3

        
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/5

         
        self.convblock13 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, dilation = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 32/5

        # TRANSITION BLOCK 1
        self.transBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            #nn.BatchNorm2d(64),
        ) # output_size = 32/5
        #self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16/6/2

        # CONVOLUTION BLOCK 2
        self.convblock21 = nn.Sequential(
            #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            #Use depth wise convolution instead of regular convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32,padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1,padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/10/2

        self.convblock22 = nn.Sequential(
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            #Use depth wise convolution instead of regular convolution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64,padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1,padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/10/2

        self.convblock23 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, dilation = 3,  bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/10/2

         # TRANSITION BLOCK 2
        self.transBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 32/5
        #self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16/6/2

         # CONVOLUTION BLOCK 3

        self.convblock31 = nn.Sequential(
            #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            #Use depth wise convolution instead of regular convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32,padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1,padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/14/2

        self.convblock32 = nn.Sequential(
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),
            #Use depth wise convolution instead of regular convolution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64,padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1,padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/14/2

        self.convblock33 = nn.Sequential(
            nn.Conv2d(in_channels=64 , out_channels=64, kernel_size=(3, 3), padding=0, dilation = 4, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/14/2

        # TRANSITION BLOCK 3
        self.transBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16/14/2
        
         # CONVOLUTION BLOCK 4

        self.convblock41 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, dilation=1, bias=False),           
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 8/24/4
        self.convblock42 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 6/32/4

        self.convblock43 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, dilation = 4, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 4/40/4
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.transBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        #print('shape of x before convolution = ', x.shape)
        #C1
        x = self.convblock11(x)
        x = self.convblock112(x)
        x = self.convblock113(x)
        #print('Shape after C11 = ',x.shape)
        x = x + self.convblock12(x)
        #print('Shape after C12 = ',x.shape)
        x = self.convblock13(x)
        #print('Shape after C13 = ',x.shape)

        #T1
        x = self.transBlock1(x)

        #C2
        x = self.convblock21(x)
        #print('Shape after C21 = ',x.shape)
        x = x + self.convblock22(x)
        #print('Shape after C22 = ',x.shape)
        x = self.convblock23(x)
        #print('Shape after C23 = ',x.shape)

        #T2
        x = self.transBlock2(x)

        #C3
        x = self.convblock31(x)
        #print('Shape after C31 = ',x.shape)
        x = x + self.convblock32(x)
        #print('Shape after C32 = ',x.shape)
        x = self.convblock33(x)
        #print('Shape after C33 = ',x.shape)

        #T3
        x = self.transBlock3(x)

        #C4
        x = self.convblock41(x)
        #print('Shape after C41 = ',x.shape)
        x = x + self.convblock42(x)
        #print('Shape after C42 = ',x.shape)
        x = self.convblock43(x)
        #print('Shape after C43 = ',x.shape)

        
        #T4
        x = self.gap(x)    
        #print('Shape after gap = ',x.shape)    
        x = self.transBlock4(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
def getModel():
    return Net().to(utils.getDevice())
