# [⭐️ CVPR 2024 Highlight ⭐️] LeGO: Leveraging a Surface Deformation Network for Animatable Stylized Face Generation with One Example
[![arXiv](https://img.shields.io/badge/arXiv-2201.12345-brightgreen?style=flat&logo=arXiv)](https://arxiv.org/abs/2403.15227)
[![](https://img.shields.io/badge/project-page-red.svg)](https://kwanyun.github.io/lego/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Open-ff69b4?style=flat&logo=googlecolab)](https://colab.research.google.com/drive/17PWpoy-UruGDTPum_MYZ-sa6dhGlXXQM?usp=sharing)



### [[Project Page]](<https://kwanyun.github.io/lego/>) [[arXiv]](<https://arxiv.org/abs/2403.15227>) [[Paper]](paper/LeGO_CVPR2024.pdf) 

![Teaser Image](readme_images/teaser.png)

## Overview
Recent advances in 3D face stylization have made significant strides in few-shot to zero-shot settings. However, the degree of stylization achieved by existing methods often falls short for practical applications because they primarily rely on statistical 3D Morphable Models (3DMM) with limited variations. To address this, we propose a method capable of producing highly stylized 3D face models with desired topology. Our approach trains a surface deformation network with 3DMM and translates its domain to the target style using a paired exemplar. Stylization of the 3D face mesh is achieved by mimicking the style of the target using a differentiable renderer and directional CLIP losses. Additionally, during inference, we utilize a Mesh Agnostic Encoder (MAGE) that takes a deformation target, a mesh with diverse topologies, as input to the stylization process and encodes its shape into our latent space. The resulting stylized face model can be animated using commonly used 3DMM blend shapes. We demonstrate our method's effectiveness through a series of quantitative and qualitative evaluations, showcasing its ability to produce highly stylized face meshes according to a given style and output them in desired topologies. We also present example applications of our method, including image-based stylized avatar generation, linear interpolation of geometric styles, and facial animation of stylized avatars.


![Teaser Image](readme_images/method.png)
## Features
- **Pretraining DS**: Trained to generate versatile head meshes with different shapes and expressions using the FLAME model.
- **Fine-tuning DT**: Implements one-shot stylization schemes for 3D face meshes via domain adaptation and hierarchical rendering.
- **Training Mesh Agnostic Encoder (MAGE)**: Encodes diverse mesh topologies into a topology-invariant latent space for stylization.


## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-repo/LeGO_3D_Stylization.git
cd LeGO_3D_Stylization
```

2. **Install Dependencies**
```bash
bash ./require.sh
```

3. **Required Downloads**
- MICA components:
  - Clone [MICA repository](https://github.com/Zielon/MICA.git)
  - Download [mica.py](https://drive.google.com/file/d/1p0HTYdYCJTuonMiOMK2BB7m1wA-7qoq7/view?usp=drive_link)
  - Download [models folder](https://drive.google.com/drive/folders/1pkEPgCqMm6jW1OaR_Op_3PdVCaA8_xbH?usp=sharing)
- [FLAME2020](https://flame.is.tue.mpg.de/) (place in `data/`)
- Checkpoints:
  - [Pretrained DS model](https://drive.google.com/drive/folders/1II18BGnK65hY54ATc26LaSAOlReejqHk?usp=sharing)
  - [Pretrained DT styles](https://drive.google.com/drive/folders/1II18BGnK65hY54ATc26LaSAOlReejqHk?usp=sharing)

## Usage

### Generate Stylized Face
```bash
python inference.py --config ./configs/inference.yml --styles <style_name>
```

### Generate with Custom Topology
```bash
python inference_desired_topology.py --config ./configs/inference_desired_topology.yml --styles <style_name> --topology <topology_type>
```
Available topologies: `mask`, `simplified`, `FLAME`, `looped_modified`

### Generate with Expression
```bash
python animation.py --config ./configs/animation.yml --styles <style_name> --expression <expression_type>
```
Available expressions: `sad`, `happy`, `angry`, `surprised`

### Fine-tune Style
```bash
python oneshot_train.py --config ./configs/fine-tuning.yml --name_data <style_name>
```

## Citation
```bibtex
@article{yoon2024lego,
  title={LeGO: Leveraging a Surface Deformation Network for Animatable Stylized Face Generation with One Example},
  author={Yoon, Soyeon and Yun, Kwan and Seo, Kwanggyoon and Cha, Sihun and Yoo, Jung Eun and Noh, Junyong},
  journal={arXiv preprint arXiv:2403.15227},
  year={2024}
}
```

## Contact
- Soyeon Yoon: thoyeony@kaist.ac.kr
- Kwan Yun: yunandy@kaist.ac.kr

## Acknowledgments
This work builds upon [Dif-net](https://github.com/microsoft/DIF-Net), [DD3C](https://github.com/ycjungSubhuman/DeepDeformable3DCaricatures/tree/main), and [Text2Mesh](https://github.com/threedle/text2mesh).
