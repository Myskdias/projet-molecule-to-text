import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss for contrastive learning with cosine similarity.
    Adapted for 1-positive-per-anchor (diagonal) setup.
    """
    def __init__(self, gamma=32, m_pos=0.25, m_neg=0.25):
        super().__init__()
        self.gamma = gamma
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.softplus = nn.Softplus()

    def forward(self, sim_matrix: torch.Tensor):
        """
        sim_matrix: cosine similarity matrix [B, B]
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device

        # positives: diagonal
        sp = torch.diag(sim_matrix)  # [B]

        # negatives: off-diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        sn = sim_matrix[mask].view(B, B - 1)

        # weighting factors
        alpha_p = torch.relu(-sp + 1 + self.m_pos)
        alpha_n = torch.relu(sn + self.m_neg)

        # margins
        delta_p = 1 - self.m_pos
        delta_n = self.m_neg

        # logits
        logit_p = -self.gamma * alpha_p * (sp - delta_p)
        logit_n = self.gamma * alpha_n * (sn - delta_n)

        # aggregate
        loss_p = self.softplus(torch.logsumexp(logit_p, dim=0))
        loss_n = self.softplus(torch.logsumexp(logit_n, dim=1)).mean()

        loss = loss_p + loss_n
        return loss
