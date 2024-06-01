import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import igl
import yaml
import tqdm
import io
import numpy as np
import time
import dataset_2 as dataset
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
import math
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from surface_deformation import create_mesh, create_mesh_single, create_mesh_2

from MICA.mica import MICA
from dataset_2 import Normalize
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


import random

#======= Modules for One-shot =======
from oneshot_modules_aug import *
import oneshot_loss 
from text2mesh.opt import ClipOpt
#===================================


def calculate_normals(vertex_V, surface_F):
    v1 = vertex_V[surface_F[:, 0]]
    v2 = vertex_V[surface_F[:, 1]]
    v3 = vertex_V[surface_F[:, 2]]
    cross_product = torch.cross(v2 - v1, v3 - v1)
    cross_product /= cross_product.clone().norm(dim = -1, keepdim = True).to(device)
    
    return cross_product

def get_obj(V, F, name):
    V = V[0].cpu().numpy()
    F = F.cpu().numpy()
    #F = np.array(F)
    #F = torch.from_numpy(F).to(device)
    igl.write_obj(f'./saved_result/{name}.obj', V, F)


    
def train(
    fixed_model, learnable_model, dataset, device,lrecon,l1,l2,l3,lreg,l1_mode,l2_mode, name_data,checkpoint_path, epochs, lr, steps_til_summary, 
    epochs_til_checkpoint, model_dir, summary_dir,num_clip=10,start_num_clip = 0, single_img = False, close_up = False,
    loss_schedules=None, is_train=True, V_ref=None, F=None, landmarks=None, dims_lmk=3,normal_mode = 'ours', **kwargs, ):

    print('Training Info:')
    print('batch_size:\t\t',kwargs['batch_size'])
    print('epochs:\t\t\t',epochs)

    print('learning rate:\t\t',lr)
    print(f'{name_data} {lrecon} {l1} {l2} {lreg} {l1_mode} {l2_mode}')
    for key in kwargs:
        if 'loss' in key:
            print(key+':\t',kwargs[key])
            
       
    if is_train:#for training 
        checkpoint = torch.load(checkpoint_path)
        optim = torch.optim.Adam(lr = lr, params=learnable_model.parameters())
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler = StepLR(optimizer = optim, step_size=200, gamma=0.9)
            
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)
    
    #model_dir = os.path.join(model_dir, f"{name_data}lrec123regenc_{lr}_{lrecon}_{l1}_{l2}_{l3}_{lreg}_{l2_mode}")
    if single_img: 
        num_clip = 1
    if close_up:
        start_num_clip = 6
        
        
    model_dir = os.path.join(model_dir, f"{name_data}_normal_{lreg}_normal_mode_{normal_mode}_img_{single_img}_level_{close_up}")

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    mesh_dir = os.path.join(model_dir, 'meshes')
    utils.cond_mkdir(mesh_dir)

    writer = SummaryWriter(summary_dir)
    
    #Define loss function 
    cosSim = torch.nn.CosineSimilarity(dim=1)
    MSE = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    #1. Get reference vector from normal to stylized
    normal_path = f'./train_data/normal/{name_data}.obj'
    optim_flamecode = Optimization(fixed_model, dataset.vertices, dataset.faces, device)
    optim_flamecode.optimize(mesh_path = normal_path)
    flame_code_ref = optim_flamecode.flame_code.to(device)
    V_ref = optim_flamecode.V.to(device)
 
    get_obj(V_ref, dataset.faces, 'recon')
    
    
    
    
    
    stylized_path = f'./train_data/stylized/{name_data}_style.obj'
    stylized_mesh = trimesh.load(stylized_path)
    stylized = Normalize().normalize(torch.tensor(stylized_mesh.vertices).to(device), torch.tensor(stylized_mesh.faces))    
    stylized = torch.tensor(stylized).float().to(device)
    #get_obj(stylized, dataset.faces, 'stylized')
    
    #Initialize the clip and Rendering model 
    clipmodel = 'ViT-B/32'
    clip_option = ClipOpt()
    
    name_data = name_data+f'_{num_clip}'
    if single_img or close_up: 
        name_data = name_data

        
    
    
    
    output_dir=os.path.join("./oneshot_output", f"{name_data}_normal_{lreg}_normal_mode_{normal_mode}_img_{single_img}_level_{close_up}")

    utils.cond_mkdir(output_dir)
    rendering_clip = Rendering_CLIP(clipmodel,dataset.vertices, dataset.faces, device, output_dir,clip_option.get_opt_dict())
    dataset.vertices= torch.tensor(dataset.vertices).float().to(device)
    surfaces_flame = torch.tensor(dataset.faces).to(device)
    normal_ref_style = calculate_normals(stylized,surfaces_flame)
    
    
    mica_checkpoint = './mica.tar'
    mica_model = MICA(mica_checkpoint, device)
    normalization = Normalize()
    
    
    total_steps = 0
    iters = kwargs['iters']
    with tqdm(total=iters*epochs) as pbar:
        train_losses = []
        losses = {}
        LAMDA_RECON = lrecon
        LAMDA_1 = l1
        LAMDA_2 = l2
        LAMDA_3 = l3
        LAMDA_REG=lreg
        
        facial_part_lamdas = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0,
                                           1.0, 1.0 ,1.0, 1.0, 1.0]).to(device)
        

        for epoch in range(epochs):
            
            torch.save({'model': learnable_model.state_dict(),
                                        'optimizer':optim.state_dict()},
                                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            for step in range(iters):
                
                if l2_mode == "Linear":
                    LAMDA_2 = l2*step/(iters * epochs)
                elif l2_mode == "Binary":
                    if step > epochs*iters/2:
                        LAMDA_2 = l2
                    else:
                        LAMDA_2 = 0
                elif l2_mode == "Full":
                    LAMDA_2 = l2
                elif l2_mode == "Zero":
                    LAMDA_2 = 0
                else:
                    assert True, "Mode Error"
                    
                start_time = time.time()
                
                torch.manual_seed(step+1e10)
                obj_name = str(step)
                
                #1. References CLIP Embedding
                                
                normal_embedding = rendering_clip.get_clip_embedding(V_ref, 'normal', step = step, epoch = epoch)[start_num_clip:num_clip]
                stylized_embeddings = rendering_clip.get_clip_embedding(stylized, 'stylized', step = step, epoch = epoch)[start_num_clip:num_clip]
                
                #Subtract vectors
                VECTOR_REF = (stylized_embeddings - normal_embedding)

                #2. Recontruction loss (Stylized, Stylized')
                stylized_ref_intermediate = learnable_model.module.inference_back(dataset.vertices, flame_code_ref).to(device)

                recon_loss = torch.nn.functional.mse_loss(stylized, stylized_ref_intermediate.squeeze(0))*LAMDA_RECON
                
                losses['RECON_LOSS'] = recon_loss
                
                #3. Cosine similarity loss(CLIP loss) (Stylized, Stylized')
                ## Rendering & clip stylized_intermediate 
                stylized_ref_intermediate_embeddings = rendering_clip.get_clip_embedding(stylized_ref_intermediate, 'flame_code_ref_stylized', step = step, epoch = epoch, obj_name = obj_name)[start_num_clip:num_clip]
                
                ## Calculate the cosine similarity between stylized and stylized_intermediate 
                cosine_sim_loss = MSE(stylized_embeddings,stylized_ref_intermediate_embeddings)
                
                cosine_sim_loss*= LAMDA_1
            
                losses['COSINE_SIM_LOSS_1_front_face'] = cosine_sim_loss 
                                                
                #4. Flamecode Samples
                expression_codes = torch.normal(0, 0.8, size=(1, 100)).to(device)
                shape_codes = torch.normal(0, 0.8, size=(1, 300)).to(device)
                flame_code_sample = torch.cat((shape_codes, expression_codes), axis = -1)
                
                with torch.no_grad():
                    sample_normal_V = fixed_model.module.inference_back(dataset.vertices,flame_code_sample)
                
                ## Infer mesh using flame_code_sample : Learnable model
                stylized_sample_intermediate = learnable_model.module.inference_back(dataset.vertices, flame_code_sample).to(device)
                
                ## Render & Clip sample_normal_V
                sample_normal_embeddings = rendering_clip.get_clip_embedding(sample_normal_V, 'flame_code_sample_normal', step = step, epoch = epoch, obj_name = obj_name)[start_num_clip:num_clip]
                ## Render & Clip stylized_sample_intermediate
                sample_stylized_intermediate_embeddings= rendering_clip.get_clip_embedding(stylized_sample_intermediate, 'flam_code_sample_stylized', step = step, epoch = epoch, obj_name = obj_name)[start_num_clip:num_clip]
                # Get V_stylized
                Vector_stylized = sample_stylized_intermediate_embeddings - stylized_embeddings                 
                # Get V_sample
                Vector_sample = sample_stylized_intermediate_embeddings - sample_normal_embeddings
                # Get V_normal 
                Vector_normal = sample_normal_embeddings - normal_embedding
                                        
                #5. Calculate the cosine similarity
                    #torch.nn.functional.mse_loss
                cosine_sim_loss2 = MSE(VECTOR_REF, Vector_sample)

                cosine_sim_loss2*=LAMDA_2
                
                cosine_sim_loss3 = MSE(Vector_normal,Vector_stylized)
                cosine_sim_loss3*=LAMDA_3
                
                if normal_mode == 'ours':
                    flame_code_sample_exp = torch.cat((shape_codes, flame_code_ref[:,300:]), axis = -1)
                    stylized_sample_ref_exp = learnable_model.module.inference_back(dataset.vertices, flame_code_sample_exp).to(device)
                    normal_sample_style = calculate_normals(stylized_sample_ref_exp.squeeze(0),surfaces_flame)
                    
                
                elif normal_mode == 'original':
                    stylized_sample_exp = learnable_model.module.inference_back(dataset.vertices, flame_code_sample).to(device)
                    normal_sample_style = calculate_normals(stylized_sample_exp.squeeze(0),surfaces_flame)
                    
                
                else: 
                    assert True, "Mode Error"
    
                cos_normal_loss = (1. - cosSim(normal_ref_style,normal_sample_style))*LAMDA_REG
                
                
                losses['COSINE_SIM_LOSS_2'] = cosine_sim_loss2
                losses['COSINE_SIM_LOSS_3'] = cosine_sim_loss3
                losses['SURFACE_NORMAL_LOSS'] = cos_normal_loss
                if is_train:
                    losses = losses
                else:
                    losses = model.embedding(embedding, model_input,gt, landmarks, dims_lmk)

                fixed_params = list(fixed_model.parameters())
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss
                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                '''
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(loss_list), newline='\n')
                '''
                
                if not total_steps % steps_til_summary:
                    if is_train:
                        print(f"RECON_LOSS : {recon_loss:.7f}\t Clip_1:  {cosine_sim_loss.item():.7f}\t Clip_2:  {cosine_sim_loss2:.7f}\t Clip_3:  {cosine_sim_loss3:.7f}\t normal:  {cos_normal_loss.mean():.5f}")
                
                
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)
                total_steps += 1
                #scheduler.step()
                
            tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
            #tqdm.write("R_loss %d, Cosine_sim_1_loss %d, Cosine_sim_2_loss %d, Cosine_sim_3_loss %d" % (loss_list[0], loss_list[1], loss_list[2], loss_list[3]))
                


            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                    np.array(train_losses))

        

if __name__ == "__main__":
    
    
    p = configargparse.ArgumentParser()
    p.add_argument('--config',  default='./configs/oneshot_train.yml',is_config_file=True, help='Evaluation configuration')   
    p.add_argument('--model_dir', type=str, default='./train/logs', help='root for logs')
    p.add_argument('--summary_dir', type=str, default='./train_1/summaries', help='root for summary')
    p.add_argument('--experiment_name', type=str, default='default',
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    # General training options
    p.add_argument('--batch_size', type=int, default=256, help='training batch size.')
    p.add_argument('--lr', type=float, default=3e-5, help='learning rate. default=1e-4')
    p.add_argument('--epochs', type=int, default=0, help='Number of epochs to train for.')

    p.add_argument('--epochs_til_checkpoint', type=int, default=3,
                help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=100,
                help='Time interval in seconds until tensorboard summary is saved.')

    p.add_argument('--model_type', type=str, default='sine',
                help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

    p.add_argument('--latent_dim', type=int,default=128, help='latent code dimension.')
    p.add_argument('--hidden_num', type=int,default=128, help='hidden layer dimension of deform-net.')
    p.add_argument('--num_hidden_layers', type=int,default=3, help='number of hidden layers of deform-net.')
    p.add_argument('--hyper_hidden_layers', type=int,default=1, help='number of hidden layers hyper-net.')
    p.add_argument('--use_abs', type=bool,default=True, help='Use absolute position for fc')
    p.add_argument('--num_train', type=int, default =1000,help='Specify the number of training data')
    p.add_argument('--checkpoint_path', type=str, help = 'Specify the checkpoint path where you start the training')
    
    #params
    p.add_argument('--lrecon', type=float,default=120, help = 'Specify the params')
    p.add_argument('--l1', type=float,default=8, help = 'Specify the params')
    p.add_argument('--l2', type=float,default=0.5, help = 'Specify the params')
    p.add_argument('--l3', type=float,default=5, help = 'Specify the params')
    p.add_argument('--lreg', type=float,default=0, help = 'Specify the params')
    p.add_argument('--l1_mode', type=str,default='CLIP', help = 'Specify encoder mode')
    p.add_argument('--l2_mode', type=str,default='Linear', help = 'Specify encoder mode')
    p.add_argument('--num_clip', type=int,default=10, help = 'name of one shot data')
    p.add_argument('--name_data', type=str,default='boris', help = 'name of one shot data')
    p.add_argument('--latent_dim_shape', type=int,default=300, help='latent code dimension.')
    p.add_argument('--latent_dim_expression', type=int,default=100, help='latent code dimension.')
    p.add_argument('--num_samples', type=int,default=3941, help='Number of samples.')
    p.add_argument('--iters', type=int,default=1000, help='Number of iterations.')
    p.add_argument('--normal_mode', type=str,default='ours', help='Normal option.')
    p.add_argument('--start_num_clip', type=int,default=0)
    p.add_argument('--close_up', type=bool,default=False)
    p.add_argument('--single_img', type=bool,default=False)
    
    # load configs
    opt = p.parse_args()
    meta_params = vars(opt)
    
    num_train = opt.num_train #number of dataset used for refinement training 
    device = torch.device("cuda:0")
    # Load fixed model 
    fixed_model = SurfaceDeformationField(device = device, **meta_params) #Deformation NW (Input: (Nx3), Output: Displacement)
    fixed_model = nn.DataParallel(fixed_model)
    checkpoint = torch.load(meta_params['checkpoint_path'])
    fixed_model.load_state_dict(checkpoint['model'])
    
    
 
    
    if hasattr(fixed_model.module, 'hyper_net'):
        for param in fixed_model.module.hyper_net.parameters():
            param.requires_grad = False
    
    for param in fixed_model.module.flame_shape_encoder.parameters():
        param.requires_grad = False
        
    for param in fixed_model.module.flame_expression_encoder.parameters():
        param.requires_grad = False
 
            
    ## Gpu configuration 

    fixed_model.to(device)
    
    # Define learnable model 
    learnable_model = SurfaceDeformationField(device = device, **meta_params) #Deformation NW (Input: (Nx3), Output: Displacement)
    learnable_model = nn.DataParallel(learnable_model)
    learnable_model.load_state_dict(checkpoint['model'])

    
    if hasattr(learnable_model.module, 'hyper_net'):
        for param in learnable_model.module.hyper_net.parameters():
            param.requires_grad = True
   
    for param in learnable_model.module.flame_shape_encoder.parameters():
        param.requires_grad = False
        
    for param in learnable_model.module.flame_expression_encoder.parameters():
        param.requires_grad = False
    
    
            
    ## Gpu configuration 
    
    learnable_model.to(device)
    
    print('CUDA Index :', torch.cuda.current_device())
    print('GPU 이름 :', torch.cuda.get_device_name())
    print('GPU 개수 :', torch.cuda.device_count())
    
    #dir = './finetuning_data'
    dir = None
    # Define dataloader 
    dataset = dataset.TrainData(meta_params['num_train'], meta_params['num_samples'], device = device)
    train(fixed_model,learnable_model ,dataset , device, **meta_params )
    