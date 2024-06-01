import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import igl
import yaml
import tqdm
import io
import numpy as np
import time
import trimesh
from scipy.io import savemat
from scipy.spatial.transform import Rotation
import utils_dd3c, loss, modules, meta_modules
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from cyobj.io import write_obj, read_obj
import skimage.io as sio

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from surface_net_4 import SurfaceDeformationField

from render import MeshRenderer
import math

from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from surface_deformation import create_mesh_2, create_mesh_single, create_mesh_with_flame_code, create_mesh_with_flame_code_pdc

from MICA.mica import MICA
from dataset_2 import Normalize
from torch.optim.lr_scheduler import StepLR
from preprocessing import remove_eyeball 
from oneshot_modules_aug import Optimization
import pickle
from get_facial_mask import get_facial_vertices



        
def train(
    fixed_model , test_model, device, lr,
    loss_schedules=None, is_train=True, V_ref=None, F=None, save_dir = None, landmarks=None, dims_lmk=3, **kwargs, ):

    for key in kwargs:
        if 'loss' in key:
            print(key+':\t',kwargs[key])
    

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok = True)
        
    
    
    inference_mode = kwargs['inference_mode']
    
   
    #FLAME head template 
    
    '''
 
  
    template_head_path = './staticdata/facial_mesh.obj'
    template_head_mesh = trimesh.load(template_head_path)
    template_name = template_head_path.split('/')[-1]
    template_vertices = torch.tensor(template_head_mesh.vertices).float().to(device)
    template_faces = torch.tensor(template_head_mesh.faces).to(device)
    '''
    
    template_head_path = './staticdata/eyeball_removed_head_template_normalized.obj'
    template_head_mesh = trimesh.load(template_head_path)
    
    
  
    
    template_vertices = template_head_mesh.vertices
    template_faces = template_head_mesh.faces

    '''
    # Create a new vertex array with the selected vertices
    looped_template_vertices, looped_template_faces = igl.loop(template_vertices, template_faces)
    looped_template_vertices, looped_template_faces = igl.loop(looped_template_vertices, looped_template_faces)
    #template_vertices = template_vertices[::4, :]
    '''
    
    
    template_faces = torch.tensor(template_faces).to(device)
    template_vertices = torch.tensor(template_vertices).float().to(device)
    
    


 

  
  
   
   
    '''
    #Looped head template 
    
    new_V_ref_loop_1, loop_1_fids = igl.loop(template_vertices.clone().cpu().numpy(), template_faces.clone().cpu().numpy())
    looped_vertices, looped_faces = igl.loop(new_V_ref_loop_1, loop_1_fids )
    looped_vertices = torch.tensor(looped_vertices).float().to(device)
    looped_faces = torch.tensor(looped_faces).to(device)
    '''
    
   
    


    
  
    

        
    mica_checkpoint = './mica.tar'
    mica_model = MICA(mica_checkpoint, device)

    
    #Start inference 
    dtype = torch.float32
    
    ckpt_root_path = './ablation_result'
    #ckpts = os.listdir(ckpt_root_path)
    ckpts = ['full']
    
    styles = ['19', 'boris', 'carell', 'disney']
    #styles = [ '15', '16',  'boris', 'disney']

    test_obj_paths = kwargs['test_paths']
    test_pts = os.listdir(test_obj_paths)
    
    with tqdm(total=len(ckpts)*len(styles)*len(test_objs)) as pbar:
        for ckpt in ckpts: 

            for style in styles:
                ckpt_path = os.path.join(ckpt_root_path, f'{ckpt}/{style}/checkpoints/model_epoch_0001.pth')
                result_dir = os.path.join(save_dir, f'{ckpt}/{style}')
                
                if not os.path.isdir(result_dir):
                    os.makedirs(result_dir, exist_ok = True)

                checkpoint_stylized = torch.load(ckpt_path)
                test_model.load_state_dict(checkpoint_stylized['model'])




                for idx, test_pt in enumerate(test_pts): 
                    if not test_obj.endswith(".pt"):
                        continue
                    obj_path = os.path.join(test_obj_paths, test_obj)
                 
                    #V_test, F_test = igl.read_triangle_mesh(obj_path)
                    import pdb; pdb.set_trace()
                    #V_test = torch.tensor(V_test[::4,:]).float().to(device)
                    #test_vertices, _ = get_facial_vertices(V_test)
                    
                    '''
                    looped_test_vertices, looped_test_faces = igl.loop(V_test, F_test)
                    looped_test_vertices, looped_test_faces = igl.loop(looped_test_vertices, looped_test_faces)
                    looped_test_vertices = torch.tensor(looped_test_vertices).float().to(device)
    
               
                    
           
                   '''
                    #Original
                    optim_flame_code = Optimization(fixed_model,template_vertices, template_faces, device)
                    optim_flame_code.optimize(mesh_V = looped_test_vertices)
                    V_test_optimized = optim_flame_code.V.clone().detach().squeeze()
                    test_flame_code_optimized = optim_flame_code.flame_code.clone().detach()
                    
                    

                    
                    '''

                    #Looped
                    test_V_loop_1, test_loop_1_fids = igl.loop(test_vertices.cpu().numpy(), test_faces.cpu().numpy())
                    looped_test_vertices, looped_test_faces = igl.loop(test_V_loop_1, test_loop_1_fids)

                    looped_test_vertices = torch.tensor(looped_test_vertices).float().to(device)
                    looped_test_faces = torch.tensor(looped_test_faces).to(device)

                    optim_flame_code_looped = Optimization(fixed_model,looped_vertices, looped_faces, device)
                    optim_flame_code_looped.optimize(mesh_V = looped_test_vertices)
                    V_test_looped_optimized = optim_flame_code_looped.V.clone().detach().squeeze()
                    test_looped_flame_code_optimized = optim_flame_code_looped.flame_code.clone().detach()
                    '''

                    
                   
                    
                    
                 

                    name = os.path.splitext(test_obj)[0]
                    
                    #igl.write_obj(os.path.join(result_dir,  f'optimized_{name}_FLAME.obj'), V_test_optimized.cpu().numpy(), test_faces.cpu().numpy())
                    igl.write_obj(os.path.join(result_dir, f'optimized_{name}_looped.obj'), looped_test_vertices.cpu().numpy(), template_faces.cpu().numpy())
                    #igl.write_obj(os.path.join(result_dir,  f'optimized_{name}_simplified.obj'), V_test_simplified_optimized.cpu().numpy(), simplified_faces_test.cpu().numpy())


                    '''
                    create_mesh_with_flame_code(
                                test_model, 
                                os.path.join(result_dir, f'inference_{name}.obj'),
                                template_vertices.float(),
                                template_faces,
                                flame_code = test_flame_code_optimized
                                )
                    '''
                    create_mesh_with_flame_code_pdc(
                                test_model, 
                                os.path.join(result_dir, f'inference_{name}_looped.obj'),
                                template_vertices.float(),
                                template_faces,
                                flame_code = test_flame_code_optimized
                        
                    )
                   
                
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
    p.add_argument('--checkpoint_stylized_path', type=str, default = None, help = 'Specify the checkpoint path where you start the training')
    p.add_argument('--checkpoint_fixed_path', type=str, default = None, help = 'Specify the checkpoint path where you start the training')
    p.add_argument('--inference_mode', type=int, help = 'Specify the stylization mode' )
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


    train(fixed_model, test_model, device, **meta_params )
