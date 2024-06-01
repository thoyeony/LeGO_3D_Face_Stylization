import os
import numpy as np 
import igl 
import trimesh

flame_mask = np.load('./FLAME2020/FLAME_masks/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')

def remove_eyeball(save_dir = None, input_dir = None, input_mesh = None):
    
    
    left_eyeball = flame_mask['left_eyeball']
    right_eyeball = flame_mask['right_eyeball']
    both_eyeball = np.concatenate((left_eyeball, right_eyeball))
    if input_dir is not None: 
        mesh_list = os.listdir(input_dir)
    
        for idx, mesh in enumerate(mesh_list): 
            v, f = igl.read_triangle_mesh(os.path.join(input_dir, mesh))
            v =np.delete(v, both_eyeball, axis = 0)
            
            if idx == 0: 
                updated_faces = []
                for face in f:
                    if any(vertex_index in both_eyeball for vertex_index in face):
                        continue
                    updated_faces.append(face)
                igl.write_obj(os.path.join(save_dir, mesh), v, np.array(updated_faces))
            else: 
                igl.write_obj(os.path.join(save_dir, mesh), v, np.array(updated_faces))
    else:
        v =np.delete(input_mesh[0], both_eyeball, axis = 0)
        return np.array(v)
 
        

   


if __name__ == "__main__":
    save_dir = './inference_output'
    os.makedirs(save_dir, exist_ok = True)
    remove_eyeball(dir, save_dir)
    
    
    
    