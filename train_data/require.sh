apt-get update -y
apt-get upgrade -y
pip install numpy
pip install matplotlib
pip install h5py
pip install pyrender
pip install trimesh[easy]
pip install scipy==1.9.3
pip install cyobj
pip install tqdm
pip install scikit_image
pip install scikit_learn
pip install ConfigArgParse
pip install tensorboard
pip install PyYaml
pip install torchmeta
pip install setuptools==59.5.0
pip install pickle5
pip install lpips
pip install pytorch3d
pip install opencv-python
pip install insightface
pip install onnxruntime
pip install loguru
pip install chumpy==0.70

apt-get upgrade libstdc++6
apt-get install libglfw3-dev libgles2-mesa-dev -y
conda install -c conda-forge igl=2.2.0 -y
#pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

pip install git+https://github.com/openai/CLIP.git@3702849800aa56e2223035bccd1c6ef91c704ca8
#pip install git+https://github.com/NVIDIAGameWorks/kaolin@a00029e5e093b5a7fe7d3a10bf695c0f01e3bd98
#pip install git+https://github.com/NVIDIAGameWorks/kaolin@75ca02ce49b02cfdb1e92764296556283375ca73
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.10.1_cu113.html