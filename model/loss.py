import torch as t
from torch import nn

class Myloss(nn.Module):

    def __init__(self):
        super(Myloss, self).__init__()

    # Eq.(11)
    def forward(self, target, prediction, dis_lap, sno_lap, alpha1, alpha2, sizes):
        loss_ls = t.norm((target - prediction), p='fro') ** 2 # Calculate least squares loss (Frobenius norm of the difference)
        dis_reg = t.trace(t.mm(t.mm(alpha1.T, dis_lap), alpha1)) # Calculate regularization term for diseases and snoRNAs
        sno_reg = t.trace(t.mm(t.mm(alpha2.T, sno_lap), alpha2))      
        graph_reg = sizes.lambda1 * dis_reg + sizes.lambda2 * sno_reg # Combine disease and snoRNA regularization terms
        loss_sum = loss_ls + graph_reg # Total loss
        return loss_sum.sum()
