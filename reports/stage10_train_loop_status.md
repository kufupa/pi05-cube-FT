# phase10_train_loop
Executed: 2026-04-04T12:39:34Z

## StageA (real only)
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
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/lerobot-train --policy.type smolvla --policy.pretrained_path jadechoghari/smolvla_metaworld --policy.load_vlm_weights true --policy.vlm_model_name HuggingFaceTB/SmolVLM2-500M-Instruct --policy.expert_width_multiplier 0.5 --policy.self_attn_every_n_layers 0 --policy.n_action_steps 1 --dataset.root /vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged/train --policy.max_steps 6000 --policy.report_to none --policy.output_dir /vol/bitbucket/aa6622/pi05-cube-FT/artifacts/stage10
```

## StageB (JMPA mix)
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
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/lerobot-train --policy.type smolvla --policy.pretrained_path jadechoghari/smolvla_metaworld --policy.load_vlm_weights true --policy.vlm_model_name HuggingFaceTB/SmolVLM2-500M-Instruct --policy.expert_width_multiplier 0.5 --policy.self_attn_every_n_layers 0 --policy.n_action_steps 1 --dataset.root /vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged/val --policy.max_steps 6000 --policy.report_to none --policy.output_dir /vol/bitbucket/aa6622/pi05-cube-FT/artifacts/stage10/jepa_mix
```
SMOLVLA_ENABLE_VGG not set; skipping StageC.

- [PASS] stage loop driver executed
  - 2026-04-04T12:39:34Z
