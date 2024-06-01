# LeGO: Leveraging a Surface Deformation Network for Animatable Stylized Face Generation with One Example

![Teaser Image](path_to_teaser_image/teaser_image.png)

## Overview
Recent advances in 3D face stylization have made significant strides in few to zero-shot settings. However, the degree of stylization achieved by existing methods is often not sufficient for practical applications because they are mostly based on statistical 3D Morphable Models (3DMM) with limited variations. To this end, we propose a method that can produce a highly stylized 3D face model with desired topology. Our methods train a surface deformation network with 3DMM and translate its domain to the target style using a paired exemplar. The network achieves stylization of the 3D face mesh by mimicking the style of the target using a differentiable renderer and directional CLIP losses. Additionally, during the inference process, we utilize a Mesh Agnostic Encoder (MAGE) that takes deformation target, a mesh of diverse topologies as input to the stylization process and encodes its shape into our latent space. The resulting stylized face model can be animated by commonly used 3DMM blend shapes. A set of quantitative and qualitative evaluations demonstrate that our method can produce highly stylized face meshes according to a given style and output them in a desired topology. We also demonstrate example applications of our method including image-based stylized avatar generation, linear interpolation of geometric styles, and facial animation of stylized avatars.

## Features 
- **Pretraining DS**: Trained to generate versatile head meshes with different shapes and expressions using the FLAME model. This network can manipulate global head shapes and local expressions through shape parameters, allowing the creation of diverse face meshes.
- **Fine-tuning DT**: Implements one-shot stylization schemes for 3D face meshes via domain adaptation and hierarchical rendering. By using a paired exemplar (identity and style), the network is fine-tuned to produce highly stylized 3D faces while preserving the original identity.
- **Training Mesh Agnostic Encoder (MAGE)**: Encodes diverse mesh topologies into a topology-invariant latent space for stylization. This allows the method to handle input meshes with various topologies and project them into a latent space for consistent stylization.

## DS Checkpoints
The pretrained checkpoint of the source face deformation network (DS) can be downloaded from the following link:
- [Pretrained Checkpoint for DS](https://drive.google.com/file/d/1GTQ90hhn09QEtBMQyQ_iuH7NVOrlia5P/view?usp=drive_link)

## Setup Instructions

### Requirements
- Python 3.8 or later 
- PyTorch 1.10.1
- Kaolin 0.13.0
- Other dependencies as specified in `require.sh`

### Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/LeGO_3D_Stylization.git
    cd LeGO_3D_Stylization
    ```

2. **Install Dependencies**:
    ```bash
    bash ./require.sh
    ```

3. **Download Checkpoints**:
    - Download the pretrained DS checkpoint from the link provided above and place it in the `ckpt` directory.
  
4. **Download MICA**:
    - In the training process, we use flame module from [MICA]([https://drive.google.com/file/d/1GTQ90hhn09QEtBMQyQ_iuH7NVOrlia5P/view?usp=drive_link](https://github.com/Zielon/MICA)) directory. 

## Usage

### Fine-tuning DT
To finetune DT for stylization:
```bash
python main.py --train --config configs/train_config.yaml
 
