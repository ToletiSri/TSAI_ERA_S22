def dice_loss(pred, target):
    smooth = 1e-5
    
    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice  


import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        pred = pred.float()
        target = target.float()
        pred.requires_grad = True
        target.requires_grad = True

        # Define constants for classes
        FOREGROUND = 1
        BACKGROUND = 2
        NOT_CLASSIFIED = 3

        # Convert trimaps to binary masks for foreground
        mask_pred_foreground = (pred == FOREGROUND).float()  # Binary mask: 1 for foreground, 0 otherwise
        mask_target_foreground = (target == FOREGROUND).float()  # Binary mask: 1 for foreground, 0 otherwise

        mask_pred_foreground.requires_grad = True
        mask_target_foreground.requires_grad = True

        # Calculate intersection and union for the foreground
        intersection = torch.sum(mask_pred_foreground * mask_target_foreground, dim=(1, 2, 3))
        union = torch.sum(mask_pred_foreground + mask_target_foreground, dim=(1, 2, 3))  # Union = A + B - Intersection

        
        # Flatten predictions and targets
        # pred = pred.view(-1)
        # target = target.view(-1)
        
        # intersection = (pred * target).sum()
        # union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice