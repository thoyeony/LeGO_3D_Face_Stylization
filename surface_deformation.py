import igl
import numpy as np
import time
import torch
import trimesh

def create_mesh(model, filename, V_ref, F, embedding=None):
    """
    From trained model and embeddings, create meshes
    """
    device = embedding.device
    coords = V_ref.to(device)
    V = model.module.inference(coords, embedding)
    V = V.cpu().numpy()[0]
    F = F.cpu().numpy()
    #F = np.array(F)
    #F = torch.from_numpy(F).to(device)
    igl.write_obj(filename, V, F)
    #save_obj(filename, V[0], F)

def create_mesh_2(model, filename, V_ref, F, flame_dict=None, device = None):
    """
    From trained model and embeddings, create meshes
    """
    coords = V_ref.to(device)
    
    V = model.module.inference(coords, flame_dict)
    
    V = V.cpu().numpy()[0]
    F = F.cpu().numpy()
    #F = np.array(F)
    #F = torch.from_numpy(F).to(device)
    igl.write_obj(filename, V, F)
    #save_obj(filename, V[0], F)

def create_mesh_single(model, filename, V_ref, F, embedding=None):
    """
    From trained model and embeddings, create meshes
    """
    device = embedding.device
    coords = V_ref.to(device)
    V = model.inference(coords, embedding)
    #V = V.cpu().numpy()[0]
    F = np.array(F)
    F = torch.from_numpy(F)
    mesh = torch.meshgrid(V, F)
    #igl.write_obj(filename, V, F)
    save_obj(filename, mesh)
    
def create_mesh_with_flame_code(model, filename, V_ref, F, flame_code=None, device = None):
    """
    From trained model and embeddings, create meshes
    """
    coords = V_ref.to(device)
    
    
    V = model.module.inference_with_flame_code(coords, flame_code)
    
    V = V.cpu().numpy()[0]
    F = F.cpu().numpy()
    #F = np.array(F)
    #F = torch.from_numpy(F).to(device)
    igl.write_obj(filename, V, F)

def create_mesh_with_flame_code_pdc(model, filename, V_ref, F, flame_code=None, device = None):
    """
    From trained model and embeddings, create meshes
    """
    coords = V_ref.to(device)
    
    V = model.module.inference_with_flame_code(coords, flame_code)
    
    V = V.cpu().numpy()[0]
    F = F.cpu().numpy()
    #F = np.array(F)
    #F = torch.from_numpy(F).to(device)
    cloud = trimesh.points.PointCloud(V)
    cloud.export(filename)

