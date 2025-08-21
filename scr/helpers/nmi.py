## Normalized Mutual Information (NMI)
## input predicted tensors and ground truth tensors (ints)

from torch import Tensor
import torch

def nmi(pred: Tensor, target: Tensor) -> float:
    # Compute the Normalized Mutual Information (NMI) between the predicted and target tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute the joint histogram
    joint_hist = torch.histc(pred * target, bins=100, min=0, max=1)

    # Compute the marginal histograms
    pred_hist = torch.histc(pred, bins=100, min=0, max=1)
    target_hist = torch.histc(target, bins=100, min=0, max=1)

    # Compute the joint and marginal entropies
    joint_entropy = -torch.sum(joint_hist * torch.log(joint_hist + 1e-10))
    pred_entropy = -torch.sum(pred_hist * torch.log(pred_hist + 1e-10))
    target_entropy = -torch.sum(target_hist * torch.log(target_hist + 1e-10))

    # Compute the NMI
    nmi = 2 * (joint_entropy - pred_entropy - target_entropy) / (pred_entropy + target_entropy)

    return nmi.item()