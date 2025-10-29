# src/training/synthetic/losses.py

import torch
import torch.nn as nn

class DictL2Loss(nn.Module):
    """
    Calculates the L2 (MSE) loss between two dictionaries of tensors.
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_dict: dict, gt_dict: dict) -> torch.Tensor:
        """
        Args:
            pred_dict (dict): Dictionary of predicted tensors.
            gt_dict (dict): Dictionary of ground truth tensors.

        Returns:
            torch.Tensor: The total loss, averaged over all fields.
        """
        total_loss = 0.0
        num_fields = 0
        
        # We iterate over the *ground truth* keys to ensure we only
        # compute loss on fields we have a target for.
        for key in gt_dict.keys():
            if key not in pred_dict:
                print(f"Warning: Ground truth key '{key}' not in prediction dict.")
                continue
                
            pred = pred_dict[key]
            gt = gt_dict[key]
            total_loss = total_loss + self.mse_loss(pred, gt)
            num_fields += 1
            
        return total_loss / num_fields if num_fields > 0 else 0.0