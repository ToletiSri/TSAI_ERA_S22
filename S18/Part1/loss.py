#def dice_loss(pred, target):
#    smooth = 1e-5
    
#    # flatten predictions and targets
#    pred = pred.view(-1)
#    target = target.view(-1)
    
#    intersection = (pred * target).sum()
#    union = pred.sum() + target.sum()
    
#    dice = (2. * intersection + smooth) / (union + smooth)
    
#    return 1 - dice  


import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        target = torch.cat([ (target == i) for i in range(1,4) ], dim=1)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
      
        return torch.mean(1 - dice)
