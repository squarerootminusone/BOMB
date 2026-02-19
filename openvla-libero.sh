# Create environment
conda create -n openvla python=3.10 -y
conda activate openvla

# PyTorch (adjust cuda version as needed)
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Core scientific packages via conda (faster, better dependency resolution)
conda install numpy=1.26.4 scipy scikit-learn joblib numba llvmlite cloudpickle -c conda-forge -y

# OpenGL / rendering deps for LIBERO/MuJoCo
conda install glfw pyopengl mujoco -c conda-forge -y

# Image/video
conda install imageio opencv -c conda-forge -y

# NLP/misc
conda install nltk future -c conda-forge -y

# Jupyter-related
conda install traitlets jupyter_core nbformat jsonschema -c conda-forge -y

# Install OpenVLA (pip needed for editable installs)
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# Install LIBERO
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# Install remaining LIBERO requirements that aren't on conda
cd ../openvla
pip install -r experiments/robot/libero/libero_requirements.txt --no-deps

# Pin specific versions for reproducibility
conda install transformers=4.40.1 tokenizers=0.19.1 timm=0.9.10 -c conda-forge -y

# Fix numpy if it got overwritten
conda install numpy=1.26.4 -c conda-forge -y
