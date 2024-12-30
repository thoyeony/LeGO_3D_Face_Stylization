
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import igl
import yaml
import tqdm
import io
import numpy as np
import time
from dataset import Normalize 
from scipy.io import savemat
from scipy.spatial.transform import Rotation
import loss, modules, meta_modules

import matplotlib.pyplot as plt
import scipy as sp
from cyobj.io import write_obj, read_obj
import lpips

import skimage.io as sio
import torch
from torch.utils.data import DataLoader
import configargparse
import trimesh
from torch import nn
from surface_net import SurfaceDeformationField

#=================CLIP Modules================#
import clip
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
from rendering_module.render import Renderer
from rendering_module.mesh import Mesh
from rendering_module.Normalization import MeshNormalizer
import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
#=============================================#
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from surface_deformation import create_mesh, create_mesh_single, create_mesh_2

class AlternatingOptimizer_Flame:
    def __init__(self, optim_flame_code, iters=3):
        
        self.optim_flame_code = optim_flame_code
        self.flame_code = torch.zeros(1,400).float()
        self.iters = iters

    
 
        

    def run(self):
        #Optimize Z
        for _ in range(self.iters):
            self.flame_code = self.optim_flame_code.run(self.flame_code)
           
        
        self.V = self.optim_flame_code.decode(self.flame_code)
        
        
class OptimizerBaseline:
    def __init__(self, model, 
            V_ref, F,
            lr=3e-3, criteria=1e-10, device=None,
            max_iter=100000,V_model=None):
        
        self.device = device
        self.model = model.to(self.device)
        self.lr = lr
        self.criteria = criteria
        self.max_iter = max_iter
        self.V_ref = V_ref.to(self.device)
        self.F = F
        self.V_model = V_model
 

    def decode(self, flame_code):
        result = self.model.module.inference_with_flame_code(self.V_ref, flame_code)
        return result

    def run(self,flame_init):
        flame_code = flame_init.to(self.device).requires_grad_(True)
        opt = torch.optim.Adam([flame_code], lr=self.lr)

      
        F = self.F.long().to(self.device)
        error_prev = 1000000000
        tolerance_for_cari = 2
        tolerance_for_others = 1
        curr_tol =0
       
        for i in range(self.max_iter):
            opt.zero_grad()
            V_new = self.model.module.inference_back(self.V_ref, flame_code)

            
                        
            loss = 2e4*torch.nn.functional.mse_loss(V_new.squeeze(), self.V_model)
            loss += torch.mean((flame_code.squeeze()[:300])**2)*0.1 + torch.mean((flame_code.squeeze()[300:])**2)*0.5
            loss.backward()
            opt.step()

            error = loss.item()
            delta = error_prev - error
            if delta / (error+1e-6) < self.criteria:
                curr_tol+=1
            else:
                curr_tol=0
            if curr_tol>tolerance_for_cari:
                print(loss.item())
                print('Early_cari\titer : {}'.format(i))
                break
            error_prev = loss.item()
        return flame_code 


class Optimization: #Return optimized z and the vertices of the optmized mesh 
    def __init__(self, model, V_ref, faces, device):
        self.device = device
        self.model = model.to(self.device)
        self.V = V_ref
        self.faces = faces

        
    def get_mesh(self, mesh_path = None, mesh_V = None):
        if mesh_path is not None: 
            mesh = trimesh.load(mesh_path)
            mesh = Normalize().normalize(torch.tensor(mesh.vertices).to(self.device), torch.tensor(mesh.faces).to(self.device))   
            mesh_V = mesh.float()
            
        return mesh_V
    
    def optimize(self, mesh_path=None, mesh_V = None):
        if mesh_path is not None: 
            mesh_V = self.get_mesh(mesh_path)
        optim_flame_code = OptimizerBaseline(
            model=self.model, 
            V_ref=self.V,
            F=self.faces,
            device = self.device,
            V_model = mesh_V
            )
        alter = AlternatingOptimizer_Flame(optim_flame_code)
        alter.run()
        self.flame_code = alter.flame_code.to(self.device)
        self.V = alter.V

       
        
        

class Rendering_CLIP:
    
    def __init__(self, clipmodel, V_ref, F_ref, device, output_dir,args):
        self.args = args
        #self.constrain_randomness()
        self.V_ref = V_ref
        self.F_ref = F_ref
        self.device = device
        self.clip_model, self.preprocess = clip.load(clipmodel,self.device, jit=True)
        self.res = 224 
        self.n_views = 10
        # Adjust output resolution depending on model type (Image encoders)
        if clipmodel == "ViT-L/14@336px":
            self.res = 336
        if clipmodel == "RN50x4":
            self.res = 288
        if clipmodel == "RN50x16":
            self.res = 384
        if clipmodel == "RN50x64":
            self.res = 448
        
        
        self.render = Renderer(dim=(self.res, self.res))
        self.pc = -1
        
        self.background = None
        if self.args['background'] is not None:
            assert len(self.args['background']) == 3
            self.background = torch.tensor(self.args['background']).to(device)
            
        self.losses = []

        self.n_augs = self.args['n_augs']
        self.dir = output_dir
        
        self.clip_setting()
        #self.augmentation_setting()
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        
        self.default_color = torch.zeros(len(self.V_ref), 3).to(self.device)
        self.default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(self.device)

        
        
            
        
        
            
    def constrain_randomness(self):
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        torch.cuda.manual_seed_all(self.args['seed'])
        random.seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    def clip_setting(self):
        self.clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))#((mean_R, mean_G, mean_B), (sd_R, sd_G, sd_B)
        # CLIP Transform
        self.clip_transform = transforms.Compose([
            transforms.Resize((self.res, self.res)),
            self.clip_normalizer])
        

    
    def augmentation_setting(self):
        self.displaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.res, scale=(self.args['normmincrop'], self.args['normmincrop'])),
            transforms.RandomPerspective(fill=1, p=0.5, distortion_scale=0.5),
            self.clip_normalizer])
    
     
    
    def rendering(self, V, name = None, step = None):
        mesh = Mesh(V = V.squeeze(0), F = torch.tensor(self.F_ref).squeeze(0))
        MeshNormalizer(mesh)()
        
        sampled_mesh = mesh
 
        self.set_camera(step = step)
        sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(self.default_color.unsqueeze(0),
                                                                                   sampled_mesh.faces)
        geo_renders, elev, azim = self.render.render_front_views(sampled_mesh, self.n_views,
                                                                show=False,
                                                                #center_azim=args.frontview_center[1],
                                                                #center_elev=args.frontview_center[0],
                                                                center_elev=0,
                                                                center_azim=0,
                                                                std=self.args['frontview_std'],
                                                                return_views=True,
                                                                background=self.args['background'])
        return geo_renders
        
    def save_rendered_img(self, save_path, geo_renders, name, step = None, epoch = None, obj_name = 'ori'):
        if epoch is not None and  not step%100 : 
            for i, img in enumerate(geo_renders):
                saved_path = os.path.join(os.path.join(save_path,name,str(epoch)))
                try: 
                    os.makedirs(saved_path, exist_ok = True)
                except: 
                    continue
                torchvision.utils.save_image(img, os.path.join(saved_path,   str(step)+'_{}_{}.jpg'.format(os.path.splitext(obj_name)[0],i)))
        if epoch is None:  
            for i, img in enumerate(geo_renders):
                
                saved_path = os.path.join(save_path,name)
                try: 
                    os.makedirs(saved_path, exist_ok = True)
                except: 
                    continue
                torchvision.utils.save_image(img, os.path.join(saved_path,  '{}.jpg'.format(i)))
            
                
    def get_clip_embedding(self, V, name, step = None, epoch = None, obj_name = 'ori'):
        geo_renders =self.rendering(V, name = name, step= step)
        result = []
        self.save_rendered_img(self.dir, geo_renders, name, step, epoch, obj_name)
        if self.args['n_normaugs'] > 0:
            for image in geo_renders:   
                result.append(self.clip_model.encode_image(image.unsqueeze(0)).reshape(512,))
        return torch.stack(result, dim=0)
    
    def get_image_loss(self,V1,V2, name1,name2, step = None, epoch = None,method  = 'LPIPS', obj_name = 'ori'):
        geo_renders1 =self.rendering(V1, name = name1, step= step)
        geo_renders2 =self.rendering(V2, name = name2, step= step)
        loss = 0
        self.save_rendered_img(self.dir, geo_renders1, name1, step, epoch, obj_name)
        self.save_rendered_img(self.dir, geo_renders2, name2, step, epoch, obj_name)
        if self.args['n_normaugs'] > 0:
            for i in range(len(geo_renders1)):
                if method == 'LPIPS':
                    loss += self.lpips_loss(geo_renders1[i].unsqueeze(0),geo_renders2[i].unsqueeze(0)) 
                elif method == 'L1':
                    loss += self.l1_loss(geo_renders1[i],geo_renders2[i])
                else:
                    assert True, "wrong input for reconstruction"
        return loss       
        
        
    def set_camera(self, step = None):
        
        if step is not None and self.pc!=step :
    
            self.pc = step 
            self.render.multi_camera = []
            torch.manual_seed(step)
            torch.cuda.manual_seed(step)
            torch.cuda.manual_seed_all(step)
            self.render.lights = (torch.tensor([1.0, 0.0, 0.5,1.0, 0.0, 0.0, 1.0, 0.0, 0.0])+ torch.rand((9))-0.5*torch.ones((9))).unsqueeze(0).to(self.device)
            # 0. 얼굴 정면 1. 왼쪽 얼굴 2. 오른쪽 얼굴 3. 얼굴 정면 확대 4. 왼쪽 얼굴 확대 5. 오른쪽 얼굴 확대 
            # 6. 왼쪽 눈 7. 오른쪽 눈 8. 코 9. 입 
            fovy = torch.tensor([np.pi/2.5, np.pi/2.5, np.pi/2.5, np.pi/4, np.pi/4, np.pi/4, 
                                np.pi/10, np.pi/10, np.pi/10, np.pi/10])
            angle_range = torch.rand((10))-0.5*torch.ones(10)
            azim = torch.tensor([np.pi/2, 0.0, np.pi, np.pi/2, 0.0, np.pi, np.pi/2, np.pi/2, np.pi/2, np.pi/2])+angle_range/10
            
            
            dx = torch.ones(10)
            dy = torch.ones(10)
            elev = torch.rand((10))-0.5*torch.ones(10)
            look_at = torch.tensor([[0.0, 0.0, 0.0], 
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.25, -0.2, 0.0],
                                    [-0.25, -0.2, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.25, 0.0]])
            cameras_setting = torch.stack([elev, azim, fovy, dx, dy],dim = 1)
            for c, l in zip(cameras_setting,look_at) :
                self.render.set_multi_cameras(c[0], c[1], c[2], c[3], c[4], l)
            
            
            
            
        if step is None:
            fovy = torch.tensor([np.pi/3, np.pi/3, np.pi/3, np.pi/5, np.pi/4, np.pi/4, 
                                np.pi/10, np.pi/10, np.pi/10, np.pi/10])
            azim = torch.tensor([np.pi/2, 0.0, np.pi, np.pi/2, 0.0, np.pi, 
                                np.pi/2, np.pi/2, np.pi/2, np.pi/2])
            elev = torch.zeros(10)
            dx = torch.ones(10)
            dy = torch.ones(10)
            look_at = torch.tensor([[0.0, 0.0, 0.0], 
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.25, -0.2, 0.0],
                                    [-0.25, -0.2, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.25, 0.0]])
            
            cameras_setting = torch.stack([elev, azim, fovy, dx, dy],dim = 1)
            for c, l in zip(cameras_setting,look_at) :
                self.render.set_multi_cameras(c[0], c[1], c[2], c[3], c[4], l)

    
    
    
    
    

    
        
    
    