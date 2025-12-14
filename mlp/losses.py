import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import lightning as L
import torchmetrics

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=None,  # có thể là float, list hoặc tensor
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits đầu ra (batch_size, num_classes)
        targets: nhãn thật (batch_size, num_classes)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction='none')
        pt = torch.exp(-BCE_loss)

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            if alpha.ndim == 1 and alpha.shape[0] == inputs.shape[1]:
                alpha = alpha.unsqueeze(0)  # (1, num_classes)
            F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class WeightedFocalLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma: float = 2.0,
                 term_weights: torch.Tensor = None):
        """
        term_weights: Tensor shape (num_classes,) chứa trọng số IA cho từng label.
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.term_weights = term_weights

    def forward(self, inputs, targets):
        """
        inputs: logits (B, C)
        targets: labels (B, C)
        """
        # Tính Binary Cross Entropy Loss không reduction
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        
        # Tính Focal term
        if self.alpha is not None:
            # Alpha balancing (nếu dùng)
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * bce_loss
            
        # Áp dụng trọng số IA (Term Weights)
        if self.term_weights is not None:
            weights = self.term_weights.to(inputs.device)
            # weights shape: (C,) -> broadcast thành (B, C)
            weighted_loss = focal_loss * weights
            return weighted_loss.sum() / (inputs.size(0) * weights.sum())
            
        else:
            return focal_loss.mean()