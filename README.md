# LeGO: Leveraging a Surface Deformation Network for Animatable Stylized Face Generation with One Example
[![arXiv](https://img.shields.io/badge/arXiv-2201.12345-brightgreen?style=flat&logo=arXiv)](https://arxiv.org/abs/2403.15227)
![Teaser Image](readme_images/teaser.png)

## Overview
Recent advances in 3D face stylization have made significant strides in few to zero-shot settings. However, the degree of stylization achieved by existing methods is often not sufficient for practical applications because they are mostly based on statistical 3D Morphable Models (3DMM) with limited variations. To this end, we propose a method that can produce a highly stylized 3D face model with desired topology. Our methods train a surface deformation network with 3DMM and translate its domain to the target style using a paired exemplar. The network achieves stylization of the 3D face mesh by mimicking the style of the target using a differentiable renderer and directional CLIP losses. Additionally, during the inference process, we utilize a Mesh Agnostic Encoder (MAGE) that takes deformation target, a mesh of diverse topologies as input to the stylization process and encodes its shape into our latent space. The resulting stylized face model can be animated by commonly used 3DMM blend shapes. A set of quantitative and qualitative evaluations demonstrate that our method can produce highly stylized face meshes according to a given style and output them in a desired topology. We also demonstrate example applications of our method including image-based stylized avatar generation, linear interpolation of geometric styles, and facial animation of stylized avatars.

## Features 
- **Pretraining DS**: Trained to generate versatile head meshes with different shapes and expressions using the FLAME model. 
- **Fine-tuning DT**: Implements one-shot stylization schemes for 3D face meshes via domain adaptation and hierarchical rendering. 
- **Training Mesh Agnostic Encoder (MAGE)**: Encodes diverse mesh topologies into a topology-invariant latent space for stylization.



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
2. **Download MICA and Modification**:
    - In the training process, we use the FLAME module from the [MICA](https://github.com/Zielon/MICA) repository.
    ```bash
    git clone https://github.com/Zielon/MICA.git
    cd MICA
    bash ./install.sh
    ```
    - Place `mica.py` to the MICA directory provided from the following link:
    [mica.py](https://drive.google.com/file/d/1p0HTYdYCJTuonMiOMK2BB7m1wA-7qoq7/view?usp=drive_link)
    - Replace `models/` in the MICA directory with the folder provided from the following link:
      models/](https://drive.google.com/drive/folders/18lvZcKIfqb1rGdWO4-OijfikvcT7szT0?usp=drive_link)
    - Download [FLAME2020](https://flame.is.tue.mpg.de/) and place it in `LeGO_3D_Face_Stylization/`.
    - Place `mica.tar` under `LeGO_3D_Face_Stylization/`. 



3. **Install Dependencies**:
    ```bash
    bash ./require.sh
    ```

4. **Download Checkpoints**:
    - Download the pretrained DS checkpoint from the link provided above and place it in the `ckpt` directory.
  


## Usage

### Fine-tuning DT
To finetune DT for stylization:
```bash
python oneshot_train.py --config ./configs/fine-tuning.yml   --name_data <style>
``` 

## Contact

For any inquiries, please contact the authors:

- **Soyeon Yoon**: [thoyeony@kaist.ac.kr](mailto:thoyeony@kaist.ac.kr)
- **Kwan Yun**: [yunandy@kaist.ac.kr](mailto:yunandy@kaist.ac.kr)


