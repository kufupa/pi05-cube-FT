# phase07_jepa_setup
Task: push-v3
Executed: 2026-04-03T23:15:56Z

## JEPA install (pip fallback)
```
Obtaining file:///vol/bitbucket/aa6622/VGG%20JEPA/jepa-wms
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Getting requirements to build editable: started
  Getting requirements to build editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Collecting torch>=2.7.0 (from JEPA-WMs==1.0.0)
  Using cached torch-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (29 kB)
Collecting torchvision==0.22.0 (from JEPA-WMs==1.0.0)
  Downloading torchvision-0.22.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (6.1 kB)
Collecting torchrl>=0.9.2 (from JEPA-WMs==1.0.0)
  Downloading torchrl-0.11.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (49 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.3/49.3 kB 2.1 MB/s eta 0:00:00

Collecting tensordict>=0.9.1 (from JEPA-WMs==1.0.0)
  Downloading tensordict-0.11.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (10 kB)
Collecting timm>=1.0.19 (from JEPA-WMs==1.0.0)
  Downloading timm-1.0.26-py3-none-any.whl.metadata (39 kB)
Collecting numpy (from JEPA-WMs==1.0.0)
  Using cached numpy-2.4.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (6.6 kB)
Collecting pandas (from JEPA-WMs==1.0.0)
  Using cached pandas-3.0.2-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (79 kB)
Collecting h5py (from JEPA-WMs==1.0.0)
  Downloading h5py-3.16.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (3.0 kB)
Collecting einops (from JEPA-WMs==1.0.0)
  Using cached einops-0.8.2-py3-none-any.whl.metadata (13 kB)
Collecting datasets (from JEPA-WMs==1.0.0)
  Using cached datasets-4.8.4-py3-none-any.whl.metadata (19 kB)
Collecting opencv-python (from JEPA-WMs==1.0.0)
  Downloading opencv_python-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl.metadata (19 kB)
Collecting pillow (from JEPA-WMs==1.0.0)
  Using cached pillow-12.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting decord>=0.6.0 (from JEPA-WMs==1.0.0)
  Downloading decord-0.6.0-py3-none-manylinux2010_x86_64.whl.metadata (422 bytes)
Collecting imageio (from JEPA-WMs==1.0.0)
  Using cached imageio-2.37.3-py3-none-any.whl.metadata (9.7 kB)
Collecting imageio-ffmpeg (from JEPA-WMs==1.0.0)
  Using cached imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Collecting moviepy (from JEPA-WMs==1.0.0)
  Downloading moviepy-2.2.1-py3-none-any.whl.metadata (6.9 kB)
Collecting mediapy (from JEPA-WMs==1.0.0)
  Downloading mediapy-1.2.6-py3-none-any.whl.metadata (4.8 kB)
Collecting lpips>=0.1.4 (from JEPA-WMs==1.0.0)
  Downloading lpips-0.1.4-py3-none-any.whl.metadata (10 kB)
Collecting torchcodec<=0.5 (from JEPA-WMs==1.0.0)
  Downloading torchcodec-0.5-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (10 kB)
Collecting scipy (from JEPA-WMs==1.0.0)
  Using cached scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting scikit-image (from JEPA-WMs==1.0.0)
  Downloading scikit_image-0.26.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (15 kB)
Collecting matplotlib (from JEPA-WMs==1.0.0)
  Downloading matplotlib-3.10.8-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.8/52.8 kB 4.5 MB/s eta 0:00:00

Collecting seaborn (from JEPA-WMs==1.0.0)
  Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
Collecting plotly (from JEPA-WMs==1.0.0)
  Downloading plotly-6.6.0-py3-none-any.whl.metadata (8.5 kB)
Collecting termcolor (from JEPA-WMs==1.0.0)
  Using cached termcolor-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Collecting hydra-core (from JEPA-WMs==1.0.0)
  Downloading hydra_core-1.3.2-py3-none-any.whl.metadata (5.5 kB)
Collecting hydra-submitit-launcher (from JEPA-WMs==1.0.0)
  Downloading hydra_submitit_launcher-1.2.0-py3-none-any.whl.metadata (1.1 kB)
Collecting omegaconf (from JEPA-WMs==1.0.0)
  Downloading omegaconf-2.3.0-py3-none-any.whl.metadata (3.9 kB)
Collecting wandb (from JEPA-WMs==1.0.0)
  Downloading wandb-0.25.1-py3-none-manylinux_2_28_x86_64.whl.metadata (11 kB)
Collecting tqdm (from JEPA-WMs==1.0.0)
  Using cached tqdm-4.67.3-py3-none-any.whl.metadata (57 kB)
Collecting submitit (from JEPA-WMs==1.0.0)
  Downloading submitit-1.5.4-py3-none-any.whl.metadata (7.4 kB)
Collecting clusterscope (from JEPA-WMs==1.0.0)
  Downloading clusterscope-0.0.32-py3-none-any.whl.metadata (4.4 kB)
Collecting ruamel.yaml (from JEPA-WMs==1.0.0)
  Downloading ruamel_yaml-0.19.1-py3-none-any.whl.metadata (16 kB)
Collecting nevergrad (from JEPA-WMs==1.0.0)
  Downloading nevergrad-1.0.12-py3-none-any.whl.metadata (10 kB)
Collecting gym==0.23.1 (from JEPA-WMs==1.0.0)
  Downloading gym-0.23.1.tar.gz (626 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 626.2/626.2 kB 26.6 MB/s eta 0:00:00

  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting gymnasium>=1.2.0 (from JEPA-WMs==1.0.0)
  Using cached gymnasium-1.2.3-py3-none-any.whl.metadata (10 kB)
Collecting pygame>=2.6.0 (from JEPA-WMs==1.0.0)
  Downloading pygame-2.6.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
Collecting pymunk==6.8.0 (from JEPA-WMs==1.0.0)
  Downloading pymunk-6.8.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
Collecting shapely (from JEPA-WMs==1.0.0)
  Downloading shapely-2.1.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.8 kB)
Collecting pybullet (from JEPA-WMs==1.0.0)
  Downloading pybullet-3.2.7.tar.gz (80.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 80.5/80.5 MB 41.4 MB/s eta 0:00:00

  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting mujoco>=3.2.0 (from JEPA-WMs==1.0.0)
  Using cached mujoco-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (42 kB)
Collecting dm-control>=1.0.34 (from JEPA-WMs==1.0.0)
  Downloading dm_control-1.0.38-py3-none-any.whl.metadata (1.5 kB)
Collecting metaworld (from JEPA-WMs==1.0.0)
  Using cached metaworld-3.0.0-py3-none-any.whl.metadata (9.7 kB)
Collecting d4rl (from JEPA-WMs==1.0.0)
  Downloading d4rl-1.1-py3-none-any.whl.metadata (270 bytes)
Collecting cython<3.0 (from JEPA-WMs==1.0.0)
  Downloading Cython-0.29.37-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl.metadata (3.1 kB)
Collecting patchelf (from JEPA-WMs==1.0.0)
  Downloading patchelf-0.17.2.4-py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.musllinux_1_1_x86_64.whl.metadata (3.7 kB)
INFO: pip is looking at multiple versions of jepa-wms to determine which version is compatible with other requirements. This could take a while.
