import os
import torch
import trimesh
import igl
from tqdm import tqdm
from torch import nn
from surface_net import SurfaceDeformationField
import configargparse

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

def inference(fixed_model, test_model, device, lr, save_dir, topology, test_paths, styles, **kwargs):
    ckpt_root_path = './ckpt/styles'
    representations = ['FLAME', 'simplified', 'looped_modified'] 
    for style in styles:
        ckpt_path = os.path.join(ckpt_root_path, f'{style}/checkpoints/model_epoch_0001.pth')
        test_model.load_state_dict(torch.load(ckpt_path)['model'])
        
        for template_type in representations:
            result_dir = os.path.join(save_dir, f'{style}/{template_type}')
            os.makedirs(result_dir, exist_ok=True)

            test_latent_path = os.path.join(test_paths, template_type)
            input_files = [pt for pt in os.listdir(test_latent_path) if pt.endswith(".pt")]


            # Load the output topology template
            template_vertices, template_faces = load_template(topology, device)

            for input_file in tqdm(input_files, desc=f"Processing style {style}"):
                input_file_path = os.path.join(test_latent_path, input_file)
                
                with torch.no_grad():
                    # Load the latent data
                    latent_data = torch.load(input_file_path)

                    # Perform the stylization inference
                    stylized_result = test_model.module.inference_with_embeddings(
                        template_vertices, latent_data
                    )
                    stylized_vertices = stylized_result.squeeze().detach().cpu()

                    # Save the stylized output
                    output_file_path = os.path.join(result_dir, f'{os.path.splitext(input_file)[0]}_{topology}_{style}.obj')
                    igl.write_obj(output_file_path, stylized_vertices.numpy(), template_faces.cpu().numpy())


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Generate stylized face into a desired topology.")
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
    parser.add_argument('--styles', nargs='+',type=str,default=['14'],help='List of styles to use for inference',)
    parser.add_argument('--topology',type=str,default='mask',help='List of styles to use for inference',)
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