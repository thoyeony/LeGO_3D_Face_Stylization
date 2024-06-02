import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import igl
import yaml
import tqdm
import numpy as np
import time
import trimesh
from scipy.io import savemat
from scipy.spatial.transform import Rotation
import utils_lego, modules, meta_modules
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage.io as sio

import torch
import configargparse
from torch import nn
from surface_net import SurfaceDeformationField

from tqdm.autonotebook import tqdm
from MICA.mica import MICA
from dataset import Normalize
from oneshot_modules_aug import Optimization




        
def inference(
    fixed_model , test_model, device, save_dir = None, **kwargs, ):

    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok = True)
        
    
    
    
    template_head_path = './staticdata/template_FLAME.obj'
    template_head_mesh = trimesh.load(template_head_path)
    template_vertices = template_head_mesh.vertices
    template_faces = template_head_mesh.faces
    template_faces = torch.tensor(template_faces).to(device)
    template_vertices = torch.tensor(template_vertices).float().to(device)
    
    


    
    #Start inference 
    dtype = torch.float32
    
    ckpt_path = kwargs['ckpt_path']
    name_data = kwargs['name_data']
    test_obj_paths = kwargs['test_paths']
    
    test_objs = os.listdir(test_obj_paths)
    
    with tqdm(total=len(test_objs)) as pbar:
       

        result_dir = os.path.join(save_dir, f'{name_data}')
        
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok = True)

        checkpoint_stylized = torch.load(ckpt_path)
        test_model.load_state_dict(checkpoint_stylized['model'])




        for idx, test_obj in enumerate(test_objs): 

            obj_path = os.path.join(test_obj_paths, test_obj)
            test_mesh = trimesh.load(obj_path)
                        
            test_vertices = test_mesh.vertices
            test_faces = test_mesh.faces
                            
            test_vertices = torch.tensor(test_vertices).to(device)
            test_faces = torch.tensor(test_faces).float().to(device)
            
            
        
            #Original
            optim_flame_code = Optimization(fixed_model,template_vertices, template_faces, device)
            optim_flame_code.optimize(mesh_V = test_vertices)
            V_test_optimized = optim_flame_code.V.clone().detach().squeeze()
            test_flame_code_optimized = optim_flame_code.flame_code.clone().detach()
            
            

            
            
            

            name = os.path.splitext(test_obj)[0]
            
            igl.write_obj(os.path.join(result_dir,  f'optimized_{name}_FLAME.obj'), V_test_optimized.cpu().numpy(), test_faces.cpu().numpy())


            
        
            pbar.update(1)
                




if __name__ == "__main__":
    

    
    
    p = configargparse.ArgumentParser()
    p.add_argument('--config', required=True, is_config_file=True, help='Evaluation configuration')
    # General training options
    p.add_argument('--lr', type=float, default=0.00006, help='learning rate. default=1e-4')

    p.add_argument('--model_type', type=str, default='sine',
                help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

    p.add_argument('--latent_dim_shape', type=int,default=300, help='latent code dimension.')
    p.add_argument('--latent_dim_expression', type=int,default=100, help='latent code dimension.')
    p.add_argument('--hidden_num', type=int,default=128, help='hidden layer dimension of deform-net.')
    p.add_argument('--num_hidden_layers', type=int,default=3, help='number of hidden layers of deform-net.')
    p.add_argument('--hyper_hidden_layers', type=int,default=1, help='number of hidden layers hyper-net.')
    p.add_argument('--use_abs', type=bool,default=True, help='Use absolute position for fc')
    p.add_argument('--save_dir', type=str, default = None, help = 'Specify the save path where you want to save results of the inference')
    p.add_argument('--checkpoint_stylized_path', type=str, default = None, help = 'Specify the checkpoint path' )
    p.add_argument('--checkpoint_fixed_path', type=str, default = None, help = 'Specify the checkpoint path')
    p.add_argument('--test_paths', type=str, default = './test_data', help = 'Specify the test data path' )
    
    # Load configs
    opt = p.parse_args()
    meta_params = vars(opt)
    device = torch.device("cuda:0")
    
    # Load fixed model 
    fixed_model = SurfaceDeformationField(device = device, **meta_params) 
    fixed_model = nn.DataParallel(fixed_model)
    checkpoint_fixed = torch.load(meta_params['checkpoint_fixed_path'])
    fixed_model.load_state_dict(checkpoint_fixed['model'])
    
    if hasattr(fixed_model.module, 'hyper_net'):
        for param in fixed_model.module.hyper_net.parameters():
            param.requires_grad = False
    
    for param in fixed_model.module.flame_shape_encoder.parameters():
        param.requires_grad = False
        
    for param in fixed_model.module.flame_expression_encoder.parameters():
        param.requires_grad = False

    # Load stylized model
    test_model = SurfaceDeformationField(device = device, **meta_params) #Deformation NW (Input: (Nx3), Output: Displacement)
    test_model = nn.DataParallel(test_model)

    

    if hasattr(test_model.module, 'hyper_net'):
        for param in test_model.module.hyper_net.parameters():
            param.requires_grad = False
   
    for param in test_model.module.flame_shape_encoder.parameters():
        param.requires_grad = False
        
    for param in test_model.module.flame_expression_encoder.parameters():
        param.requires_grad = False
    
            
    ## Gpu configuration 
    
    test_model.to(device)
    
    print('CUDA Index :', torch.cuda.current_device())
    print('GPU 이름 :', torch.cuda.get_device_name())
    print('GPU 개수 :', torch.cuda.device_count())


    inference(fixed_model, test_model, device, **meta_params )
