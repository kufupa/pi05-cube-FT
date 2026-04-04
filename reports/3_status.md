# phase03_smolvla_install
Executed: 2026-04-03T17:24:15Z

## Base package install (pip)
```
**************************************************************************
***                                                                    ***
*** gpucluster3.doc.ic.ac.uk is solely for submitting sbatch jobs.     ***
*** Please use a lab PC for testing                                    ***
*** or installing python environments                                  ***
***                                                                    ***
*** IMPORTANT: This OS is Ubuntu 24.04                                 ***
***                                                                    ***
*** Use gpucluster2.doc.ic.ac.uk or gpucluster3.doc.ic.ac.uk           ***
*** to submit jobs (Ubuntu 24.04)                                      ***
***                                                                    ***
*** 4-cores, 8GB RAM, NO GPU on this VM, please use sbatch             ***
*** to submit scripts                                                  ***
***                                                                    ***
**************************************************************************
Requirement already satisfied: pip in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (26.0.1)
Requirement already satisfied: setuptools in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (80.10.2)
Collecting setuptools
  Using cached setuptools-82.0.1-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: wheel in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.46.3)
Requirement already satisfied: packaging>=24.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from wheel) (25.0)
Using cached setuptools-82.0.1-py3-none-any.whl (1.0 MB)
Installing collected packages: setuptools
  Attempting uninstall: setuptools
    Found existing installation: setuptools 80.10.2
    Uninstalling setuptools-80.10.2:
      Successfully uninstalled setuptools-80.10.2
Successfully installed setuptools-82.0.1
Looking in indexes: https://download.pytorch.org/whl/cu124
Requirement already satisfied: torch in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (2.6.0+cu124)
Requirement already satisfied: torchvision in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.21.0+cu124)
Requirement already satisfied: torchaudio in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (2.6.0+cu124)
Requirement already satisfied: filelock in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (3.25.2)
Requirement already satisfied: typing-extensions>=4.10.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (4.15.0)
Requirement already satisfied: networkx in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (3.6.1)
Requirement already satisfied: jinja2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (2026.2.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: triton==3.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (3.2.0)
Requirement already satisfied: setuptools in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (82.0.1)
Requirement already satisfied: sympy==1.13.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from sympy==1.13.1->torch) (1.3.0)
Requirement already satisfied: numpy in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torchvision) (2.2.6)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torchvision) (12.2.0)
Requirement already satisfied: MarkupSafe>=2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from jinja2->torch) (3.0.3)
Collecting git+https://github.com/huggingface/lerobot.git
  Cloning https://github.com/huggingface/lerobot.git to /vol/bitbucket/aa6622/.cache/tmp/pip-req-build-tilkt5gp
  Resolved https://github.com/huggingface/lerobot.git to commit 4dbbcca496226e00c2281bc0d3ea00610d8c01f7
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: datasets<5.0.0,>=4.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (4.8.4)
Collecting diffusers<0.36.0,>=0.27.2 (from lerobot==0.5.1)
  Downloading diffusers-0.35.2-py3-none-any.whl.metadata (20 kB)
Requirement already satisfied: huggingface-hub<2.0.0,>=1.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (1.9.0)
Requirement already satisfied: accelerate<2.0.0,>=1.10.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (1.13.0)
Requirement already satisfied: numpy<2.3.0,>=2.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (2.2.6)
Collecting setuptools<81.0.0,>=71.0.0 (from lerobot==0.5.1)
  Downloading setuptools-80.10.2-py3-none-any.whl.metadata (6.6 kB)
Requirement already satisfied: cmake<4.2.0,>=3.29.0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (4.1.3)
Requirement already satisfied: packaging<26.0,>=24.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (25.0)
Requirement already satisfied: torch<2.11.0,>=2.2.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (2.6.0+cu124)
Requirement already satisfied: torchcodec<0.11.0,>=0.2.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (0.10.0)
Requirement already satisfied: torchvision<0.26.0,>=0.21.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (0.21.0+cu124)
Requirement already satisfied: einops<0.9.0,>=0.8.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (0.8.2)
Requirement already satisfied: opencv-python-headless<4.14.0,>=4.9.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (4.12.0.88)
Requirement already satisfied: av<16.0.0,>=15.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (15.1.0)
Requirement already satisfied: jsonlines<5.0.0,>=4.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (4.0.0)
Requirement already satisfied: pynput<1.9.0,>=1.7.8 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (1.8.1)
Requirement already satisfied: pyserial<4.0,>=3.5 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (3.5)
Requirement already satisfied: wandb<0.25.0,>=0.24.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (0.24.2)
Requirement already satisfied: draccus==0.10.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (0.10.0)
Requirement already satisfied: gymnasium<2.0.0,>=1.1.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (1.2.3)
Requirement already satisfied: rerun-sdk<0.27.0,>=0.24.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (0.26.2)
Requirement already satisfied: deepdiff<9.0.0,>=7.0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (8.6.2)
Requirement already satisfied: imageio<3.0.0,>=2.34.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from imageio[ffmpeg]<3.0.0,>=2.34.0->lerobot==0.5.1) (2.37.3)
Requirement already satisfied: termcolor<4.0.0,>=2.4.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from lerobot==0.5.1) (3.3.0)
Requirement already satisfied: mergedeep~=1.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from draccus==0.10.0->lerobot==0.5.1) (1.3.4)
Requirement already satisfied: pyyaml~=6.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from draccus==0.10.0->lerobot==0.5.1) (6.0.3)
Requirement already satisfied: pyyaml-include~=1.4 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from draccus==0.10.0->lerobot==0.5.1) (1.4.1)
Requirement already satisfied: toml~=0.10 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from draccus==0.10.0->lerobot==0.5.1) (0.10.2)
Requirement already satisfied: typing-inspect~=0.9.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from draccus==0.10.0->lerobot==0.5.1) (0.9.0)
Requirement already satisfied: psutil in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from accelerate<2.0.0,>=1.10.0->lerobot==0.5.1) (7.2.2)
Requirement already satisfied: safetensors>=0.4.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from accelerate<2.0.0,>=1.10.0->lerobot==0.5.1) (0.7.0)
Requirement already satisfied: filelock in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (3.25.2)
Requirement already satisfied: pyarrow>=21.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (23.0.1)
Requirement already satisfied: dill<0.4.2,>=0.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (0.4.1)
Requirement already satisfied: pandas in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (3.0.2)
Requirement already satisfied: requests>=2.32.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (2.33.1)
Requirement already satisfied: httpx<1.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (0.28.1)
Requirement already satisfied: tqdm>=4.66.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (4.67.3)
Requirement already satisfied: xxhash in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (3.6.0)
Requirement already satisfied: multiprocess<0.70.20 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (0.70.19)
Requirement already satisfied: fsspec<=2026.2.0,>=2023.1.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (2026.2.0)
Requirement already satisfied: orderly-set<6,>=5.4.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from deepdiff<9.0.0,>=7.0.1->lerobot==0.5.1) (5.5.0)
Requirement already satisfied: importlib_metadata in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from diffusers<0.36.0,>=0.27.2->lerobot==0.5.1) (9.0.0)
Requirement already satisfied: regex!=2019.12.17 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from diffusers<0.36.0,>=0.27.2->lerobot==0.5.1) (2026.3.32)
Requirement already satisfied: Pillow in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from diffusers<0.36.0,>=0.27.2->lerobot==0.5.1) (12.2.0)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (3.13.5)
Requirement already satisfied: cloudpickle>=1.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gymnasium<2.0.0,>=1.1.1->lerobot==0.5.1) (3.1.2)
Requirement already satisfied: typing-extensions>=4.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gymnasium<2.0.0,>=1.1.1->lerobot==0.5.1) (4.15.0)
Requirement already satisfied: farama-notifications>=0.0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gymnasium<2.0.0,>=1.1.1->lerobot==0.5.1) (0.0.4)
Requirement already satisfied: anyio in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1.0.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (4.13.0)
Requirement already satisfied: certifi in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1.0.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (2026.2.25)
Requirement already satisfied: httpcore==1.* in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1.0.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (1.0.9)
Requirement already satisfied: idna in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1.0.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (3.11)
Requirement already satisfied: h11>=0.16 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpcore==1.*->httpx<1.0.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (0.16.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.4.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (1.4.3)
Requirement already satisfied: typer in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (0.24.1)
Requirement already satisfied: imageio-ffmpeg in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from imageio[ffmpeg]<3.0.0,>=2.34.0->lerobot==0.5.1) (0.6.0)
Requirement already satisfied: attrs>=19.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from jsonlines<5.0.0,>=4.0.0->lerobot==0.5.1) (26.1.0)
Requirement already satisfied: six in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from pynput<1.9.0,>=1.7.8->lerobot==0.5.1) (1.17.0)
Requirement already satisfied: evdev>=1.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from pynput<1.9.0,>=1.7.8->lerobot==0.5.1) (1.9.3)
Requirement already satisfied: python-xlib>=0.17 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from pynput<1.9.0,>=1.7.8->lerobot==0.5.1) (0.33)
Requirement already satisfied: networkx in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (3.6.1)
Requirement already satisfied: jinja2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (3.1.6)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (12.4.127)
Requirement already satisfied: triton==3.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (3.2.0)
Requirement already satisfied: sympy==1.13.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch<2.11.0,>=2.2.1->lerobot==0.5.1) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from sympy==1.13.1->torch<2.11.0,>=2.2.1->lerobot==0.5.1) (1.3.0)
Requirement already satisfied: mypy-extensions>=0.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typing-inspect~=0.9.0->draccus==0.10.0->lerobot==0.5.1) (1.1.0)
Requirement already satisfied: click>=8.0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (8.3.1)
Requirement already satisfied: gitpython!=3.1.29,>=1.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (3.1.46)
Requirement already satisfied: platformdirs in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (4.9.4)
Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<7,>=3.19.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (6.33.6)
Requirement already satisfied: pydantic<3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (2.12.5)
Requirement already satisfied: sentry-sdk>=2.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (2.57.0)
Requirement already satisfied: annotated-types>=0.6.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from pydantic<3->wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (0.7.0)
Requirement already satisfied: pydantic-core==2.41.5 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from pydantic<3->wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (2.41.5)
Requirement already satisfied: typing-inspection>=0.4.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from pydantic<3->wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (0.4.2)
Requirement already satisfied: charset_normalizer<4,>=2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from requests>=2.32.2->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (3.4.7)
Requirement already satisfied: urllib3<3,>=1.26 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from requests>=2.32.2->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (2.6.3)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (2.6.1)
Requirement already satisfied: aiosignal>=1.4.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (1.4.0)
Requirement already satisfied: frozenlist>=1.1.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (6.7.1)
Requirement already satisfied: propcache>=0.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (0.4.1)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (1.23.0)
Requirement already satisfied: gitdb<5,>=4.0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gitpython!=3.1.29,>=1.0.0->wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (4.0.12)
Requirement already satisfied: smmap<6,>=3.0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb<0.25.0,>=0.24.0->lerobot==0.5.1) (5.0.3)
Requirement already satisfied: zipp>=3.20 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from importlib_metadata->diffusers<0.36.0,>=0.27.2->lerobot==0.5.1) (3.23.0)
Requirement already satisfied: MarkupSafe>=2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from jinja2->torch<2.11.0,>=2.2.1->lerobot==0.5.1) (3.0.3)
Requirement already satisfied: python-dateutil>=2.8.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from pandas->datasets<5.0.0,>=4.0.0->lerobot==0.5.1) (2.9.0.post0)
Requirement already satisfied: shellingham>=1.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (1.5.4)
Requirement already satisfied: rich>=12.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (14.3.3)
Requirement already satisfied: annotated-doc>=0.0.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (0.0.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub<2.0.0,>=1.0.0->lerobot==0.5.1) (0.1.2)
Downloading diffusers-0.35.2-py3-none-any.whl (4.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.1/4.1 MB 53.1 MB/s  0:00:00

Downloading setuptools-80.10.2-py3-none-any.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 110.6 MB/s  0:00:00

Installing collected packages: setuptools, diffusers
  Attempting uninstall: setuptools
    Found existing installation: setuptools 82.0.1
    Uninstalling setuptools-82.0.1:
      Successfully uninstalled setuptools-82.0.1
  Attempting uninstall: diffusers
    Found existing installation: diffusers 0.37.1
    Uninstalling diffusers-0.37.1:
      Successfully uninstalled diffusers-0.37.1


Successfully installed diffusers-0.35.2 setuptools-80.10.2
```

## SmolVLA dependency seed (pip)
```
**************************************************************************
***                                                                    ***
*** gpucluster3.doc.ic.ac.uk is solely for submitting sbatch jobs.     ***
*** Please use a lab PC for testing                                    ***
*** or installing python environments                                  ***
***                                                                    ***
*** IMPORTANT: This OS is Ubuntu 24.04                                 ***
***                                                                    ***
*** Use gpucluster2.doc.ic.ac.uk or gpucluster3.doc.ic.ac.uk           ***
*** to submit jobs (Ubuntu 24.04)                                      ***
***                                                                    ***
*** 4-cores, 8GB RAM, NO GPU on this VM, please use sbatch             ***
*** to submit scripts                                                  ***
***                                                                    ***
**************************************************************************
Requirement already satisfied: transformers in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (5.5.0)
Requirement already satisfied: diffusers in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.35.2)
Collecting diffusers
  Downloading diffusers-0.37.1-py3-none-any.whl.metadata (20 kB)
Requirement already satisfied: accelerate in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (1.13.0)
Requirement already satisfied: safetensors in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.7.0)
Requirement already satisfied: sentencepiece in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.2.1)
Requirement already satisfied: pillow in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (12.2.0)
Requirement already satisfied: huggingface-hub<2.0,>=1.5.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (1.9.0)
Requirement already satisfied: numpy>=1.17 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (2.2.6)
Requirement already satisfied: packaging>=20.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (6.0.3)
Requirement already satisfied: regex>=2025.10.22 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (2026.3.32)
Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (0.22.2)
Requirement already satisfied: typer in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (0.24.1)
Requirement already satisfied: tqdm>=4.27 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (4.67.3)
Requirement already satisfied: filelock>=3.10.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (3.25.2)
Requirement already satisfied: fsspec>=2023.5.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.4.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (1.4.3)
Requirement already satisfied: httpx<1,>=0.23.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (0.28.1)
Requirement already satisfied: typing-extensions>=4.1.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (4.15.0)
Requirement already satisfied: anyio in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (4.13.0)
Requirement already satisfied: certifi in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.25)
Requirement already satisfied: httpcore==1.* in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (1.0.9)
Requirement already satisfied: idna in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (3.11)
Requirement already satisfied: h11>=0.16 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (0.16.0)
Requirement already satisfied: importlib-metadata in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from diffusers) (9.0.0)
Requirement already satisfied: requests in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from diffusers) (2.33.1)
Requirement already satisfied: psutil in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from accelerate) (7.2.2)
Requirement already satisfied: torch>=2.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from accelerate) (2.6.0+cu124)
Requirement already satisfied: networkx in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (3.6.1)
Requirement already satisfied: jinja2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (3.1.6)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (12.4.127)
Requirement already satisfied: triton==3.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (3.2.0)
Requirement already satisfied: setuptools in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (80.10.2)
Requirement already satisfied: sympy==1.13.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from sympy==1.13.1->torch>=2.0.0->accelerate) (1.3.0)
Requirement already satisfied: zipp>=3.20 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from importlib-metadata->diffusers) (3.23.0)
Requirement already satisfied: MarkupSafe>=2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from jinja2->torch>=2.0.0->accelerate) (3.0.3)
Requirement already satisfied: charset_normalizer<4,>=2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from requests->diffusers) (3.4.7)
Requirement already satisfied: urllib3<3,>=1.26 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from requests->diffusers) (2.6.3)
Requirement already satisfied: click>=8.2.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (8.3.1)
Requirement already satisfied: shellingham>=1.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (1.5.4)
Requirement already satisfied: rich>=12.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (14.3.3)
Requirement already satisfied: annotated-doc>=0.0.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (0.0.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->transformers) (0.1.2)
Downloading diffusers-0.37.1-py3-none-any.whl (5.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.0/5.0 MB 85.8 MB/s  0:00:00

Installing collected packages: diffusers
  Attempting uninstall: diffusers
    Found existing installation: diffusers 0.35.2
    Uninstalling diffusers-0.35.2:
      Successfully uninstalled diffusers-0.35.2
Successfully installed diffusers-0.37.1
```

## Compat shim for lerobot policies/__init__
```
```

## SmolVLA smoke import
```
lerobot: 2.6.0+cu124
smolvla import: lerobot.policies.smolvla.modeling_smolvla
has VLAFlowMatching: True
```

- [PASS] SmolVLA base stack smoke checks succeeded
  - 2026-04-03T17:25:53Z
