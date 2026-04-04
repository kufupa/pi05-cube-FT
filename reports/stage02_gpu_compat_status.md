# phase02_gpu_compat
Executed: 2026-04-03T17:04:38Z

## Torch CUDA probe
```
torch 2.6.0+cu124
cuda_available False
cuda_version 12.4
```

## MuJoCo/render smoke
```
mujoco: ModuleNotFoundError: No module named 'mujoco'
gymnasium: ModuleNotFoundError: No module named 'gymnasium'
ffmpeg: ModuleNotFoundError: No module named 'ffmpeg'
MUJOCO_GL= <unset>
```

## Headless render check
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
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'mujoco'
```

- [PASS] compatibility probes captured in /vol/bitbucket/aa6622/pi05-cube-FT/reports/stage02_gpu_compat_status.md
  - 2026-04-03T17:04:42Z
