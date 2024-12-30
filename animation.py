import os
import torch
import trimesh
import igl
from tqdm import tqdm
from torch import nn
from surface_net import SurfaceDeformationField
import configargparse
from surface_deformation import create_mesh_with_flame_code

# Function to load a template
def load_template(template_type, device):
    templates = {
        'simplified': './staticdata/template_Simplified.obj',
        'looped_modified': './staticdata/template_Looped_modified.obj',
        'FLAME': './staticdata/template_FLAME.obj',
        'mask': './staticdata/template_mask.obj',
    }
    template_path = templates.get(template_type, './staticdata/template_FLAME.obj')
    mesh = trimesh.load(template_path)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)
    return vertices, faces

def inference(fixed_model, test_model, device, lr, save_dir, test_paths, styles, expression, **kwargs):

    os.makedirs(save_dir, exist_ok=True)

    ckpt_root_path = './ckpt/styles'
    representations = ['FLAME'] 

    # Expression code initialization based on selected expression
    expression_code = torch.zeros(100,).unsqueeze(0).to(device)
    if expression == 'happy':
        expression_code[0][[0, 3, 4, 5, 6, 7, 9]] += 3.0
    elif expression == 'angry':
        expression_code[0][[0, 1, 3, 7, 8]] += 3.0
        expression_code[0][[4, 6, 9]] += -3.0
    elif expression == 'sad':
        expression_code[0][0] += -2.0
        expression_code[0][2] += 2.0
        expression_code[0][[1, 4, 7, 9]] += -2.0
        expression_code[0][6] += -3.0
        expression_code[0][9] += -4.0
    elif expression == 'surprised':
        expression_code[0][[0, 6, 8]] += -3.0
        expression_code[0][[3, 4, 5]] += 3.0
    else:
        raise ValueError(f"Unsupported expression: {expression}")

    with tqdm(total=len(styles) * len(representations)) as pbar:
        for style in styles:
            ckpt_path = os.path.join(ckpt_root_path, f'{style}/checkpoints/model_epoch_0001.pth')
            test_model.load_state_dict(torch.load(ckpt_path)['model'])

            for template_type in representations:
                result_dir = os.path.join(save_dir, f'{style}/{template_type}')
                os.makedirs(result_dir, exist_ok=True)

                test_latent_path = os.path.join(test_paths, template_type)
                test_pts = [pt for pt in os.listdir(test_latent_path) if pt.endswith(".pt")]

                template_vertices, template_faces = load_template(template_type, device)

                for test_pt in test_pts:
                    pt_path = os.path.join(test_latent_path, test_pt)
                    name = os.path.splitext(test_pt)[0].replace('.', '')


                    with torch.no_grad():
                        preds = torch.load(pt_path)
                        # Combine the expression code with the test model's code
                        expression_latent = test_model.module.flame_expression_encoder(expression_code)
                        inference_latent_code = torch.cat((preds[0][:256].unsqueeze(0), expression_latent), dim=-1)
                        # Generate the mesh with the new expression
                        stylized_vertices=test_model.module.inference_with_embeddings(template_vertices,inference_latent_code)
                        

                        igl.write_obj(
                            os.path.join(result_dir, f'{name}_stylized.obj'),
                            stylized_vertices[0].cpu().numpy(),
                            template_faces.cpu().numpy(),
                        )

                    pbar.update(1)

if __name__ == "__main__":
    # Argument parser
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', required=True, is_config_file=True, help='Evaluation configuration')
    parser.add_argument('--lr', type=float, default=0.00006, help='Learning rate')
    parser.add_argument('--latent_dim_shape', type=int, default=300, help='Latent shape code dimension')
    parser.add_argument('--latent_dim_expression', type=int, default=100, help='Latent expression code dimension')
    parser.add_argument('--hidden_num', type=int, default=128, help='Deform-net hidden layer dimension')
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='Deform-net hidden layers count')
    parser.add_argument('--hyper_hidden_layers', type=int, default=1, help='Number of hyper-net hidden layers')
    parser.add_argument('--model_type', type=str, default='sine', help='Model type: "sine" or "mixed"')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save inference results')
    parser.add_argument('--checkpoint_fixed_path', type=str, help='Path to fixed model checkpoint')
    parser.add_argument('--test_paths', type=str, default='./inference_input', help='Path to test data')
    parser.add_argument('--styles', nargs='+',type=str,default=['16'],help='List of styles to use for inference')
    parser.add_argument('--expression',type=str,default='happy',help='Expression to use for inference')
    opt = parser.parse_args()


    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load fixed model
    fixed_model = nn.DataParallel(SurfaceDeformationField(device=device, **vars(opt)))
    fixed_model.load_state_dict(torch.load(opt.checkpoint_fixed_path)['model'])
    for param in fixed_model.module.parameters():
      param.requires_grad = False
    fixed_model.to(device)

    # Load stylized model
    test_model = nn.DataParallel(SurfaceDeformationField(device=device, **vars(opt)))
    test_model.to(device)

    print(f'Using device: {torch.cuda.get_device_name()} with {torch.cuda.device_count()} GPUs')

    # Run inference
    inference(fixed_model, test_model, device, **vars(opt))
