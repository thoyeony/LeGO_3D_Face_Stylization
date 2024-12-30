import torch
from torch import nn
import modules
from meta_modules import HyperNetwork
from loss import *
from collections import OrderedDict
from flameencoder import FlameShapeCodeEncoder, FlameExpressionCodeEncoder

class SurfaceDeformationField(nn.Module):
    def __init__(self, latent_dim_shape = 300, latent_dim_expression = 100, model_type='sine', hyper_hidden_layers=1, num_hidden_layers=3, hyper_hidden_features=256,hidden_num=128, device = None, **kwargs):
        super().__init__()

        # latent code embedding for training subjects
        self.latent_dim_shape = latent_dim_shape
        self.device = device
        self.latent_dim_expression = latent_dim_expression
        #self.latent_codes = nn.Embedding(num_instances, self.latent_dim)

        #nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        # Deform-Net
        self.deform_net = modules.SingleBVPNet(type=model_type,mode='mlp', hidden_features=hidden_num, num_hidden_layers=num_hidden_layers, in_features=3,out_features=3)
        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=256+64, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.deform_net)
        self.flame_shape_encoder = FlameShapeCodeEncoder(input_size = self.latent_dim_shape, latent_size = 256).cuda()
        self.flame_expression_encoder = FlameExpressionCodeEncoder(input_size = self.latent_dim_expression, latent_size = 64).cuda()
        #print(self)
        
    '''
    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding
    '''

    # for generation
    def inference(self, coords, flame_dict):
        with torch.no_grad():
            model_in = {'coords': coords}
            embedding_shape = self.flame_shape_encoder(flame_dict['shape'])
            embedding_expression = self.flame_expression_encoder(flame_dict['expression'])
            embedding = torch.cat((embedding_shape, embedding_expression), dim = -1)
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)

            deformation = model_output['model_out']
            new_coords = coords + deformation
            return new_coords
        
    def inference_with_flame_code(self, coords, flame_code):
        with torch.no_grad():
            model_in = {'coords': coords}
            embedding_shape = self.flame_shape_encoder(flame_code.squeeze(0)[:300])
            embedding_expression = self.flame_expression_encoder(flame_code.squeeze(0)[300:])
            embedding = torch.cat((embedding_shape, embedding_expression), dim = -1)

            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)

            deformation = model_output['model_out']
            new_coords = coords + deformation
            return new_coords
    
    def inference_back(self, coords, flame_code):
        model_in = {'coords': coords}
        embedding_shape = self.flame_shape_encoder(flame_code.squeeze(0)[:300])
        embedding_expression = self.flame_expression_encoder(flame_code.squeeze(0)[300:])
        embedding = torch.cat((embedding_shape, embedding_expression), dim = -1)

        hypo_params = self.hyper_net(embedding)
        model_output = self.deform_net(model_in, params=hypo_params)

        deformation = model_output['model_out']
        new_coords = coords + deformation
        return new_coords
    
    def inference_with_embeddings(self, coords, embedding):
        model_in = {'coords': coords}
        hypo_params = self.hyper_net(embedding)
        model_output = self.deform_net(model_in, params=hypo_params)

        deformation = model_output['model_out']
        new_coords = coords + deformation
        return new_coords
    '''
    def inference_with_w(self, coords, w, params_keys):
        with torch.no_grad():
            model_in = {'coords': coords}
            #change the data structure of w to Ordereddict()
            params = OrderedDict()
            param_shapes = self.hyper_net.param_shapes
            for p, result in zip(params_keys, w):
                params[p] = result
                
            model_output = self.deform_net(model_in, params = params)
            
            deformation = model_output['model_out']
            new_coords = coords + deformation
            return new_coords

    
    
    def inference_back(self, coords, embedding):
        model_in = {'coords': coords}
        hypo_params = self.hyper_net(embedding)
        model_output = self.deform_net(model_in, params=hypo_params)

        deformation = model_output['model_out']
        new_coords = coords + deformation
        return new_coords
    
    
    def inference_back_with_w(self, coords, w, params_keys):
        model_in = {'coords': coords}
        #change the data structure of w to Ordereddict()
        params = OrderedDict()
        for p, result in zip(params_keys, w):
            params[p] = result
            
        model_output = self.deform_net(model_in, params = params)
        
        deformation = model_output['model_out']
        new_coords = coords + deformation
        return new_coords
    '''
    # for training
    def forward(self, model_input, gt):
        coords = model_input['coords'] 
        flame_dict = model_input['flame_dict']
        
        embedding_shape = self.flame_shape_encoder(flame_dict['shape'])
        embedding_expression = self.flame_expression_encoder(flame_dict['expression'])
        
        embedding = torch.cat((embedding_shape, embedding_expression), dim = -1)
        
        hypo_params = self.hyper_net(embedding)

        model_output = self.deform_net(model_input, params=hypo_params)
        displacement = model_output['model_out'].squeeze()
        V_new = coords + displacement # deform into template space

        model_out = {
            'model_in':model_output['model_in'], 
            'model_out':V_new, 
            'shape_embedding':embedding_shape, 
            'expression_embedding': embedding_expression,
            'hypo_params':hypo_params}
        
        losses = surface_deformation_pos_loss(model_out, gt)
        return losses
    
    def oneshot_output(self, model_input, z, V):

        # get network weights for Deform-net using Hyper-net 
        embedding = z
        hypo_params = self.hyper_net(embedding)

        model_output = self.deform_net(model_input, params=hypo_params)
        displacement = model_output['model_out'].squeeze()
        V_new = V + displacement # deform into template space

        model_out = {
            'model_in':model_output['model_in'], 
            'model_out':V_new, 
            'latent_vec':embedding, 
            'hypo_params':hypo_params}
        
        
        return stylized_intermediate 

    # for evaluation
    def embedding(self, embed, model_input, gt, landmarks=None, dims_lmk=3):
        coords = model_input['coords'] # 3 dimensional input coordinates
        embedding = embed
        hypo_params = self.hyper_net(embedding)

        model_output = self.deform_net(model_input, params=hypo_params)
        displacement = model_output['model_out'].squeeze()
        V_new = coords + displacement # deform into template space

        model_out = {
            'model_in':model_output['model_in'],
            'model_out':V_new[:,:dims_lmk],
            'latent_vec':embedding, 
            'hypo_params':hypo_params}
        gt['positions'] = gt['positions'][:,:dims_lmk]
        
        losses = surface_deformation_pos_loss(model_out, gt)
        return losses