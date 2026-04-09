# phase01_env_topology
Executed: 2026-04-09T00:10:18Z

## Seed env lerobot_mw_py310 ensure pip bootstrap
```
Looking in links: /vol/bitbucket/aa6622/.cache/tmp/tmp9xp0ppsg
Requirement already satisfied: pip in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (26.0.1)
```

## Seed env lerobot_mw_py310 packages (pip, https://download.pytorch.org/whl/cu124)
```
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
Requirement already satisfied: setuptools in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (80.10.2)
Requirement already satisfied: sympy==1.13.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torch) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from sympy==1.13.1->torch) (1.3.0)
Requirement already satisfied: numpy in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torchvision) (2.2.6)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from torchvision) (11.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from jinja2->torch) (3.0.3)
```

## Seed env lerobot_mw_py310 ensure pip bootstrap
```
Looking in links: /vol/bitbucket/aa6622/.cache/tmp/tmpgy6ghwwd
Requirement already satisfied: pip in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (26.0.1)
```

## Seed env lerobot_mw_py310 packages (pip)
```
Requirement already satisfied: typing_extensions in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (4.15.0)
Requirement already satisfied: transformers in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (5.5.0)
Requirement already satisfied: diffusers in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.35.1)
Requirement already satisfied: accelerate in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (1.13.0)
Requirement already satisfied: safetensors in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.7.0)
Requirement already satisfied: sentencepiece in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (0.2.1)
Requirement already satisfied: pillow in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (11.3.0)
Requirement already satisfied: pyarrow in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (23.0.1)
Requirement already satisfied: huggingface-hub<2.0,>=1.5.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (1.9.0)
Requirement already satisfied: numpy>=1.17 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (2.2.6)
Requirement already satisfied: packaging>=20.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (6.0.3)
Requirement already satisfied: regex>=2025.10.22 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (2026.4.4)
Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (0.22.2)
Requirement already satisfied: typer in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (0.24.1)
Requirement already satisfied: tqdm>=4.27 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from transformers) (4.67.3)
Requirement already satisfied: filelock>=3.10.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (3.25.2)
Requirement already satisfied: fsspec>=2023.5.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.4.3 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (1.4.3)
Requirement already satisfied: httpx<1,>=0.23.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (0.28.1)
Requirement already satisfied: anyio in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (4.13.0)
Requirement already satisfied: certifi in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.25)
Requirement already satisfied: httpcore==1.* in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (1.0.9)
Requirement already satisfied: idna in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (3.11)
Requirement already satisfied: h11>=0.16 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (0.16.0)
Requirement already satisfied: importlib_metadata in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from diffusers) (9.0.0)
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
Requirement already satisfied: zipp>=3.20 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from importlib_metadata->diffusers) (3.23.0)
Requirement already satisfied: MarkupSafe>=2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from jinja2->torch>=2.0.0->accelerate) (3.0.3)
Requirement already satisfied: charset_normalizer<4,>=2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from requests->diffusers) (3.4.7)
Requirement already satisfied: urllib3<3,>=1.26 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from requests->diffusers) (2.6.3)
Requirement already satisfied: click>=8.2.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (8.3.2)
Requirement already satisfied: shellingham>=1.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (1.5.4)
Requirement already satisfied: rich>=12.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (14.3.3)
Requirement already satisfied: annotated-doc>=0.0.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from typer->transformers) (0.0.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->transformers) (0.1.2)
```

## Seed env jepa_wms_py310 ensure pip bootstrap
```
Looking in links: /vol/bitbucket/aa6622/.cache/tmp/tmppzgu2ttd
Requirement already satisfied: pip in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (24.0)
```

## Seed env jepa_wms_py310 packages (pip, https://download.pytorch.org/whl/cu124)
```
Looking in indexes: https://download.pytorch.org/whl/cu124
Requirement already satisfied: torch in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (2.6.0+cu124)
Requirement already satisfied: torchvision in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (0.21.0+cu124)
Requirement already satisfied: torchaudio in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (2.6.0+cu124)
Requirement already satisfied: filelock in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (3.25.2)
Requirement already satisfied: typing-extensions>=4.10.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (4.15.0)
Requirement already satisfied: networkx in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (3.6.1)
Requirement already satisfied: jinja2 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (2026.2.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: triton==3.2.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (3.2.0)
Requirement already satisfied: setuptools in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (70.2.0)
Requirement already satisfied: sympy==1.13.1 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torch) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from sympy==1.13.1->torch) (1.3.0)
Requirement already satisfied: numpy in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torchvision) (2.4.3)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from torchvision) (12.1.1)
Requirement already satisfied: MarkupSafe>=2.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from jinja2->torch) (3.0.3)
```

## Seed env jepa_wms_py310 ensure pip bootstrap
```
Looking in links: /vol/bitbucket/aa6622/.cache/tmp/tmp044efzml
Requirement already satisfied: pip in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (24.0)
```

## Seed env jepa_wms_py310 packages (pip)
```
Requirement already satisfied: typing_extensions in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (4.15.0)
Requirement already satisfied: transformers in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (5.5.0)
Requirement already satisfied: huggingface-hub<2.0,>=1.5.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (1.9.0)
Requirement already satisfied: numpy>=1.17 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (2.4.3)
Requirement already satisfied: packaging>=20.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (26.0)
Requirement already satisfied: pyyaml>=5.1 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (6.0.3)
Requirement already satisfied: regex>=2025.10.22 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (2026.4.4)
Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (0.22.2)
Requirement already satisfied: typer in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (0.24.1)
Requirement already satisfied: safetensors>=0.4.3 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (0.7.0)
Requirement already satisfied: tqdm>=4.27 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from transformers) (4.67.3)
Requirement already satisfied: filelock>=3.10.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (3.25.2)
Requirement already satisfied: fsspec>=2023.5.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.4.3 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (1.4.3)
Requirement already satisfied: httpx<1,>=0.23.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (0.28.1)
Requirement already satisfied: click>=8.2.1 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from typer->transformers) (8.3.2)
Requirement already satisfied: shellingham>=1.3.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from typer->transformers) (1.5.4)
Requirement already satisfied: rich>=12.3.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from typer->transformers) (14.3.3)
Requirement already satisfied: annotated-doc>=0.0.2 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from typer->transformers) (0.0.4)
Requirement already satisfied: anyio in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (4.13.0)
Requirement already satisfied: certifi in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.25)
Requirement already satisfied: httpcore==1.* in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (1.0.9)
Requirement already satisfied: idna in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (3.11)
Requirement already satisfied: h11>=0.16 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (0.16.0)
Requirement already satisfied: markdown-it-py>=2.2.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in /vol/bitbucket/aa6622/.envs/jepa_wms_py310/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->transformers) (0.1.2)
```

## Seed env vggflow_py311 ensure pip bootstrap
```
Looking in links: /vol/bitbucket/aa6622/.cache/tmp/tmpn51byo7x
Requirement already satisfied: pip in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (24.0)
```

## Seed env vggflow_py311 packages (pip, https://download.pytorch.org/whl/cu124)
```
Looking in indexes: https://download.pytorch.org/whl/cu124
Requirement already satisfied: torch in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (2.6.0+cu124)
Requirement already satisfied: torchvision in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (0.21.0+cu124)
Requirement already satisfied: torchaudio in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (2.6.0+cu124)
Requirement already satisfied: filelock in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (3.25.2)
Requirement already satisfied: typing-extensions>=4.10.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (4.15.0)
Requirement already satisfied: networkx in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (3.6.1)
Requirement already satisfied: jinja2 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (2026.2.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (12.4.127)
Requirement already satisfied: triton==3.2.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (3.2.0)
Requirement already satisfied: setuptools in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (70.2.0)
Requirement already satisfied: sympy==1.13.1 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torch) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from sympy==1.13.1->torch) (1.3.0)
Requirement already satisfied: numpy in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torchvision) (2.4.3)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from torchvision) (12.1.1)
Requirement already satisfied: MarkupSafe>=2.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from jinja2->torch) (3.0.3)
```

## Seed env vggflow_py311 ensure pip bootstrap
```
Looking in links: /vol/bitbucket/aa6622/.cache/tmp/tmp__kn270z
Requirement already satisfied: pip in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (24.0)
```

## Seed env vggflow_py311 packages (pip)
```
Requirement already satisfied: typing_extensions in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (4.15.0)
Requirement already satisfied: transformers in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (5.5.0)
Requirement already satisfied: huggingface-hub<2.0,>=1.5.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (1.9.0)
Requirement already satisfied: numpy>=1.17 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (2.4.3)
Requirement already satisfied: packaging>=20.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (26.0)
Requirement already satisfied: pyyaml>=5.1 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (6.0.3)
Requirement already satisfied: regex>=2025.10.22 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (2026.4.4)
Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (0.22.2)
Requirement already satisfied: typer in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (0.24.1)
Requirement already satisfied: safetensors>=0.4.3 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (0.7.0)
Requirement already satisfied: tqdm>=4.27 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from transformers) (4.67.3)
Requirement already satisfied: filelock>=3.10.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (3.25.2)
Requirement already satisfied: fsspec>=2023.5.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.4.3 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (1.4.3)
Requirement already satisfied: httpx<1,>=0.23.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (0.28.1)
Requirement already satisfied: click>=8.2.1 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from typer->transformers) (8.3.2)
Requirement already satisfied: shellingham>=1.3.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from typer->transformers) (1.5.4)
Requirement already satisfied: rich>=12.3.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from typer->transformers) (14.3.3)
Requirement already satisfied: annotated-doc>=0.0.2 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from typer->transformers) (0.0.4)
Requirement already satisfied: anyio in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (4.13.0)
Requirement already satisfied: certifi in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.25)
Requirement already satisfied: httpcore==1.* in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (1.0.9)
Requirement already satisfied: idna in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (3.11)
Requirement already satisfied: h11>=0.16 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (0.16.0)
Requirement already satisfied: markdown-it-py>=2.2.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from rich>=12.3.0->typer->transformers) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in /vol/bitbucket/aa6622/.envs/vggflow_py311/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->transformers) (0.1.2)
```
## env creation status
- lerobot_mw: /vol/bitbucket/aa6622/.envs/lerobot_mw_py310
- jepa_wms: /vol/bitbucket/aa6622/.envs/jepa_wms_py310
- vggflow: /vol/bitbucket/aa6622/.envs/vggflow_py311

- [PASS] all env roots created
  - 2026-04-09T00:14:12Z
