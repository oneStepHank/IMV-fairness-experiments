import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha (float): 가중치 계수 (기본 1). 불균형이 심하면 조절.
            gamma (float): 어려운 샘플에 얼마나 집중할지 (기본 2). 클수록 어려운 문제에 더 집중함.
            reduction (str): 'mean' (평균) 또는 'sum' (합계). 보통 'mean' 사용.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: 모델의 출력 (Logits) [Batch_Size, Num_Classes]
        # targets: 정답 라벨 [Batch_Size]
        
        # 1. 일반적인 Cross Entropy Loss 계산 (reduction='none'으로 개별 Loss 구함)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. prob (확률) 계산: CrossEntropy는 LogSoftmax를 포함하므로 exp(-loss)가 확률이 됨
        prob = torch.exp(-ce_loss)
        
        # 3. Focal Loss 공식 적용: alpha * (1-prob)^gamma * ce_loss
        focal_loss = self.alpha * (1 - prob) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss