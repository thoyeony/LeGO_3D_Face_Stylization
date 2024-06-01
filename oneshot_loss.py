import torch
import torch.nn.functional as F

#Reconstruction loss 
def reconstruction_loss(stylized, stylized_intermediate):
    return torch.nn.functional.mse_loss(stylized, stylized_intermediate)

def surface_deformation_pos_loss(model_output, gt):
    gt_pos = gt['positions']
    V_new = model_output['model_out']

    data_constraint = torch.nn.functional.mse_loss(V_new, gt_pos)

    # -----------------
    return {'data_constraint': data_constraint}
