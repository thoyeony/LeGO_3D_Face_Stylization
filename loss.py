import torch
import torch.nn.functional as F



def surface_deformation_pos_loss(model_output, gt):
    gt_pos = gt['positions']
    V_new = model_output['model_out']
    

    data_constraint = torch.nn.functional.mse_loss(V_new, gt_pos)
    #contrastive_contraint = contrastive_loss(torch.flatten(V_new).unsqueeze(0), torch.flatten(gt_pos).unsqueeze(0))

    
    #embeddings_constraint = torch.mean(embeddings ** 2)
    
    # -----------------
    
    '''
    return {'data_constraint': data_constraint * 3e3, 
            'contrastive_contraint': contrastive_contraint * 1}
            
    '''    
        
    '''
    return {'data_constraint': data_constraint * 3e3, 
            'embeddings_constraint': embeddings_constraint.mean() * 1e6}
    '''    
    
    return {'data_constraint': data_constraint * 3e3}
    
