import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)  # 자동으로 모델의 디바이스로 이동
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # weight를 입력 텐서와 같은 디바이스로 이동
        if self.alpha is not None:
            weight = self.alpha.to(inputs.device)
        else:
            weight = None
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)  # 예측 확률
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss 