# phase10_train_loop
Executed: 2026-04-09T11:29:55Z

## Stage10 torch/CUDA probe
```
torch 2.6.0+cu124
cuda_available True
device NVIDIA A16
```

## StageA (real only)
```
WARN: unsupported lerobot option 'logging_steps', skipping from command
WARN: unsupported lerobot option 'save_steps', skipping from command
[train-orch] running: PYTHONPATH=/vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/compat /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/lerobot-train --policy.type smolvla --policy.pretrained_path jadechoghari/smolvla_metaworld --policy.load_vlm_weights true --policy.vlm_model_name HuggingFaceTB/SmolVLM2-500M-Instruct --policy.expert_width_multiplier 0.5 --policy.self_attn_every_n_layers 0 --policy.n_action_steps 1 --dataset.root /vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged_v30_runs/run_20260408_025416/train --dataset.repo_id lerobot/pusht --steps 6000 --policy.push_to_hub false --output_dir /vol/bitbucket/aa6622/pi05-cube-FT/artifacts/run_20260408_025416/stage10_a_20260409T113116Z_aa6622_parrot_229851_job229851/stageA/train_run --wandb.enable true --wandb.mode offline
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
INFO 2026-04-09 12:34:54 ot_train.py:197 {'batch_size': 8,
 'checkpoint_path': None,
 'cudnn_deterministic': False,
 'dataset': {'episodes': None,
             'image_transforms': {'enable': False,
                                  'max_num_transforms': 3,
                                  'random_order': False,
                                  'tfs': {'affine': {'kwargs': {'degrees': [-5.0,
                                                                            5.0],
                                                                'translate': [0.05,
                                                                              0.05]},
                                                     'type': 'RandomAffine',
                                                     'weight': 1.0},
                                          'brightness': {'kwargs': {'brightness': [0.8,
                                                                                   1.2]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'contrast': {'kwargs': {'contrast': [0.8,
                                                                               1.2]},
                                                       'type': 'ColorJitter',
                                                       'weight': 1.0},
                                          'hue': {'kwargs': {'hue': [-0.05,
                                                                     0.05]},
                                                  'type': 'ColorJitter',
                                                  'weight': 1.0},
                                          'saturation': {'kwargs': {'saturation': [0.5,
                                                                                   1.5]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'sharpness': {'kwargs': {'sharpness': [0.5,
                                                                                 1.5]},
                                                        'type': 'SharpnessJitter',
                                                        'weight': 1.0}}},
             'repo_id': 'lerobot/pusht',
             'revision': None,
             'root': '/vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged_v30_runs/run_20260408_025416/train',
             'streaming': False,
             'use_imagenet_stats': True,
             'video_backend': 'torchcodec'},
 'env': None,
 'eval': {'batch_size': 50, 'n_episodes': 50, 'use_async_envs': False},
 'eval_freq': 20000,
 'job_name': 'smolvla',
 'log_freq': 200,
 'num_workers': 4,
 'optimizer': {'betas': [0.9, 0.95],
               'eps': 1e-08,
               'grad_clip_norm': 10,
               'lr': 0.0001,
               'type': 'adamw',
               'weight_decay': 1e-10},
 'output_dir': '/vol/bitbucket/aa6622/pi05-cube-FT/artifacts/run_20260408_025416/stage10_a_20260409T113116Z_aa6622_parrot_229851_job229851/stageA/train_run',
 'peft': None,
 'policy': {'adapt_to_pi_aloha': False,
            'add_image_special_tokens': False,
            'attention_mode': 'cross_attn',
            'chunk_size': 50,
            'compile_mode': 'max-autotune',
            'compile_model': False,
            'device': 'cuda',
            'empty_cameras': 0,
            'expert_width_multiplier': 0.5,
            'freeze_vision_encoder': True,
            'input_features': {},
            'license': None,
            'load_vlm_weights': True,
            'max_action_dim': 32,
            'max_period': 4.0,
            'max_state_dim': 32,
            'min_period': 0.004,
            'n_action_steps': 1,
            'n_obs_steps': 1,
            'normalization_mapping': {'ACTION': <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
                                      'STATE': <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
                                      'VISUAL': <NormalizationMode.IDENTITY: 'IDENTITY'>},
            'num_expert_layers': -1,
            'num_steps': 10,
            'num_vlm_layers': 16,
            'optimizer_betas': [0.9, 0.95],
            'optimizer_eps': 1e-08,
            'optimizer_grad_clip_norm': 10,
            'optimizer_lr': 0.0001,
            'optimizer_weight_decay': 1e-10,
            'output_features': {},
            'pad_language_to': 'longest',
            'prefix_length': -1,
            'pretrained_path': 'jadechoghari/smolvla_metaworld',
            'private': None,
            'push_to_hub': False,
            'repo_id': None,
            'resize_imgs_with_padding': [512, 512],
            'rtc_config': None,
            'scheduler_decay_lr': 2.5e-06,
            'scheduler_decay_steps': 30000,
            'scheduler_warmup_steps': 1000,
            'self_attn_every_n_layers': 0,
            'tags': None,
            'tokenizer_max_length': 48,
            'train_expert_only': True,
            'train_state_proj': True,
            'type': 'smolvla',
            'use_amp': False,
            'use_cache': True,
            'use_delta_joint_actions_aloha': False,
            'use_peft': False,
            'vlm_model_name': 'HuggingFaceTB/SmolVLM2-500M-Instruct'},
 'rabc_epsilon': 1e-06,
 'rabc_head_mode': 'sparse',
 'rabc_kappa': 0.01,
 'rabc_progress_path': None,
 'rename_map': {},
 'resume': False,
 'save_checkpoint': True,
 'save_freq': 20000,
 'scheduler': {'decay_lr': 2.5e-06,
               'num_decay_steps': 30000,
               'num_warmup_steps': 1000,
               'peak_lr': 0.0001,
               'type': 'cosine_decay_with_warmup'},
 'seed': 1000,
 'steps': 6000,
 'tolerance_s': 0.0001,
 'use_policy_training_preset': True,
 'use_rabc': False,
 'wandb': {'add_tags': True,
           'disable_artifact': False,
           'enable': True,
           'entity': None,
           'mode': 'offline',
           'notes': None,
           'project': 'lerobot',
           'run_id': None}}
INFO 2026-04-09 12:35:03 db_utils.py:117 Logs will be synced with wandb.
INFO 2026-04-09 12:35:03 db_utils.py:118 Track this run --> None
INFO 2026-04-09 12:35:03 ot_train.py:221 Creating dataset
INFO 2026-04-09 12:35:04 eo_utils.py:108 Using video codec: libsvtav1
INFO 2026-04-09 12:35:04 ot_train.py:239 Creating policy
`torch_dtype` is deprecated! Use `dtype` instead!
Loading  HuggingFaceTB/SmolVLM2-500M-Instruct weights ...

Loading weights:   0%|          | 0/489 [00:00<?, ?it/s]
Loading weights:   0%|          | 1/489 [00:02<18:13,  2.24s/it]
Loading weights:   9%|▉         | 43/489 [00:02<00:17, 25.55it/s]
Loading weights:  14%|█▍        | 68/489 [00:03<00:14, 30.05it/s]
Loading weights:  17%|█▋        | 83/489 [00:03<00:12, 33.47it/s]
Loading weights:  19%|█▉        | 94/489 [00:03<00:10, 37.04it/s]
Loading weights:  21%|██▏       | 104/489 [00:03<00:10, 35.81it/s]
Loading weights:  23%|██▎       | 113/489 [00:04<00:10, 35.54it/s]
Loading weights:  25%|██▍       | 122/489 [00:04<00:14, 25.73it/s]
Loading weights:  28%|██▊       | 136/489 [00:04<00:09, 35.60it/s]
Loading weights:  29%|██▉       | 143/489 [00:04<00:08, 39.46it/s]
Loading weights:  31%|███       | 150/489 [00:05<00:08, 38.39it/s]
Loading weights:  34%|███▎      | 164/489 [00:05<00:06, 50.66it/s]
Loading weights:  35%|███▌      | 172/489 [00:05<00:06, 49.93it/s]
Loading weights:  37%|███▋      | 179/489 [00:05<00:08, 37.78it/s]
Loading weights:  38%|███▊      | 186/489 [00:06<00:09, 31.48it/s]
Loading weights:  40%|███▉      | 194/489 [00:06<00:08, 33.02it/s]
Loading weights:  41%|████      | 199/489 [00:06<00:10, 26.55it/s]
Loading weights:  42%|████▏     | 204/489 [00:06<00:11, 24.66it/s]
Loading weights:  43%|████▎     | 210/489 [00:07<00:09, 29.46it/s]
Loading weights:  44%|████▍     | 214/489 [00:07<00:12, 22.47it/s]
Loading weights:  46%|████▌     | 223/489 [00:07<00:10, 24.65it/s]
Loading weights:  47%|████▋     | 232/489 [00:07<00:08, 31.91it/s]
Loading weights:  49%|████▉     | 240/489 [00:07<00:07, 35.00it/s]
Loading weights:  51%|█████     | 248/489 [00:08<00:06, 37.38it/s]
Loading weights:  53%|█████▎    | 257/489 [00:08<00:05, 43.10it/s]
Loading weights:  54%|█████▍    | 266/489 [00:08<00:04, 47.50it/s]
Loading weights:  56%|█████▌    | 272/489 [00:08<00:04, 47.08it/s]
Loading weights:  57%|█████▋    | 278/489 [00:08<00:05, 39.31it/s]
Loading weights:  58%|█████▊    | 285/489 [00:08<00:04, 44.88it/s]
Loading weights:  60%|██████    | 295/489 [00:09<00:03, 55.25it/s]
Loading weights:  65%|██████▍   | 317/489 [00:09<00:02, 79.72it/s]
Loading weights:  70%|███████   | 343/489 [00:09<00:01, 110.66it/s]
Loading weights:  73%|███████▎  | 357/489 [00:09<00:01, 114.80it/s]
Loading weights:  78%|███████▊  | 381/489 [00:09<00:00, 139.15it/s]
Loading weights:  81%|████████  | 397/489 [00:09<00:00, 106.98it/s]
Loading weights:  84%|████████▍ | 413/489 [00:10<00:00, 91.11it/s] 
Loading weights:  88%|████████▊ | 429/489 [00:10<00:00, 98.54it/s]
Loading weights:  91%|█████████ | 445/489 [00:10<00:00, 107.12it/s]
Loading weights:  93%|█████████▎| 457/489 [00:10<00:00, 90.89it/s] 
Loading weights:  98%|█████████▊| 477/489 [00:10<00:00, 99.34it/s]
Loading weights: 100%|██████████| 489/489 [00:10<00:00, 45.86it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING 2026-04-09 12:35:20 ls/_http.py:916 Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING 2026-04-09 12:35:50 ies/utils.py:94 Unexpected key(s) when loading model: ['model.vlm_with_expert.vlm.model.text_model.layers.16.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.16.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.17.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.18.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.19.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.20.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.21.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.22.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.23.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.24.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.25.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.26.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.27.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.28.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.29.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.30.self_attn.v_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.input_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.mlp.down_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.mlp.gate_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.mlp.up_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.post_attention_layernorm.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.self_attn.k_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.self_attn.o_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.self_attn.q_proj.weight', 'model.vlm_with_expert.vlm.model.text_model.layers.31.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.16.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.16.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.17.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.17.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.18.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.18.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.19.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.19.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.20.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.20.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.21.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.21.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.22.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.22.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.23.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.23.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.24.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.24.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.25.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.25.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.26.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.26.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.27.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.27.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.28.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.28.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.29.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.29.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.30.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.30.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.31.self_attn.k_proj.weight', 'model.vlm_with_expert.lm_expert.layers.31.self_attn.v_proj.weight', 'model.vlm_with_expert.lm_expert.layers.16.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.16.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.16.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.16.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.16.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.16.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.16.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.17.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.17.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.17.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.17.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.17.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.17.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.17.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.18.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.18.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.18.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.18.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.18.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.18.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.18.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.19.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.19.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.19.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.19.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.19.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.19.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.19.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.20.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.20.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.20.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.20.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.20.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.20.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.20.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.21.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.21.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.21.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.21.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.21.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.21.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.21.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.22.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.22.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.22.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.22.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.22.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.22.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.22.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.23.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.23.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.23.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.23.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.23.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.23.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.23.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.24.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.24.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.24.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.24.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.24.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.24.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.24.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.25.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.25.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.25.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.25.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.25.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.25.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.25.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.26.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.26.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.26.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.26.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.26.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.26.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.26.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.27.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.27.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.27.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.27.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.27.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.27.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.27.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.28.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.28.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.28.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.28.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.28.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.28.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.28.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.29.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.29.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.29.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.29.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.29.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.29.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.29.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.30.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.30.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.30.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.30.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.30.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.30.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.30.self_attn.q_proj.weight', 'model.vlm_with_expert.lm_expert.layers.31.input_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.31.mlp.down_proj.weight', 'model.vlm_with_expert.lm_expert.layers.31.mlp.gate_proj.weight', 'model.vlm_with_expert.lm_expert.layers.31.mlp.up_proj.weight', 'model.vlm_with_expert.lm_expert.layers.31.post_attention_layernorm.weight', 'model.vlm_with_expert.lm_expert.layers.31.self_attn.o_proj.weight', 'model.vlm_with_expert.lm_expert.layers.31.self_attn.q_proj.weight']
INFO 2026-04-09 12:35:52 ot_train.py:306 Creating optimizer and scheduler
INFO 2026-04-09 12:35:52 hedulers.py:105 Auto-scaling LR scheduler: num_training_steps (6000) < num_decay_steps (30000). Scaling warmup: 1000 → 200, decay: 30000 → 6000 (scale factor: 0.200)
INFO 2026-04-09 12:35:52 ot_train.py:341 Output dir: /vol/bitbucket/aa6622/pi05-cube-FT/artifacts/run_20260408_025416/stage10_a_20260409T113116Z_aa6622_parrot_229851_job229851/stageA/train_run
INFO 2026-04-09 12:35:52 ot_train.py:348 cfg.steps=6000 (6K)
INFO 2026-04-09 12:35:52 ot_train.py:349 dataset.num_frames=1400 (1K)
INFO 2026-04-09 12:35:52 ot_train.py:350 dataset.num_episodes=7
INFO 2026-04-09 12:35:52 ot_train.py:353 Effective batch size: 8 x 1 = 8
INFO 2026-04-09 12:35:52 ot_train.py:354 num_learnable_params=48284512 (48M)
INFO 2026-04-09 12:35:52 ot_train.py:355 num_total_params=398449696 (398M)
INFO 2026-04-09 12:35:52 ot_train.py:419 Start offline training on a fixed dataset, with effective batch size: 8
Reducing the number of VLM layers to 16 ...
INFO 2026-04-09 12:39:40 ot_train.py:451 step:200 smpl:2K ep:8 epch:1.14 loss:2.137 grdn:4.207 lr:5.0e-05 updt_s:0.439 data_s:0.699
INFO 2026-04-09 12:43:25 ot_train.py:451 step:400 smpl:3K ep:16 epch:2.29 loss:1.399 grdn:2.172 lr:9.9e-05 updt_s:0.368 data_s:0.754
INFO 2026-04-09 12:47:07 ot_train.py:451 step:600 smpl:5K ep:24 epch:3.43 loss:1.392 grdn:1.955 lr:9.8e-05 updt_s:0.368 data_s:0.743
INFO 2026-04-09 12:50:52 ot_train.py:451 step:800 smpl:6K ep:32 epch:4.57 loss:1.401 grdn:1.867 lr:9.7e-05 updt_s:0.367 data_s:0.757
INFO 2026-04-09 12:54:39 ot_train.py:451 step:1K smpl:8K ep:40 epch:5.71 loss:1.387 grdn:1.714 lr:9.5e-05 updt_s:0.363 data_s:0.770
INFO 2026-04-09 12:58:24 ot_train.py:451 step:1K smpl:10K ep:48 epch:6.86 loss:1.351 grdn:1.761 lr:9.2e-05 updt_s:0.370 data_s:0.749
INFO 2026-04-09 13:02:05 ot_train.py:451 step:1K smpl:11K ep:56 epch:8.00 loss:1.356 grdn:1.689 lr:8.9e-05 updt_s:0.366 data_s:0.739
INFO 2026-04-09 13:05:55 ot_train.py:451 step:2K smpl:13K ep:64 epch:9.14 loss:1.347 grdn:1.609 lr:8.6e-05 updt_s:0.367 data_s:0.780
INFO 2026-04-09 13:09:39 ot_train.py:451 step:2K smpl:14K ep:72 epch:10.29 loss:1.346 grdn:1.584 lr:8.2e-05 updt_s:0.367 data_s:0.754
INFO 2026-04-09 13:13:24 ot_train.py:451 step:2K smpl:16K ep:80 epch:11.43 loss:1.342 grdn:1.572 lr:7.8e-05 updt_s:0.367 data_s:0.753
INFO 2026-04-09 13:17:08 ot_train.py:451 step:2K smpl:18K ep:88 epch:12.57 loss:1.335 grdn:1.458 lr:7.3e-05 updt_s:0.367 data_s:0.750
INFO 2026-04-09 13:20:53 ot_train.py:451 step:2K smpl:19K ep:96 epch:13.71 loss:1.317 grdn:1.415 lr:6.9e-05 updt_s:0.367 data_s:0.757
INFO 2026-04-09 13:24:37 ot_train.py:451 step:3K smpl:21K ep:104 epch:14.86 loss:1.326 grdn:1.525 lr:6.4e-05 updt_s:0.367 data_s:0.751
INFO 2026-04-09 13:28:21 ot_train.py:451 step:3K smpl:22K ep:112 epch:16.00 loss:1.322 grdn:1.464 lr:5.9e-05 updt_s:0.366 data_s:0.752
INFO 2026-04-09 13:32:09 ot_train.py:451 step:3K smpl:24K ep:120 epch:17.14 loss:1.299 grdn:1.474 lr:5.4e-05 updt_s:0.367 data_s:0.772
INFO 2026-04-09 13:35:56 ot_train.py:451 step:3K smpl:26K ep:128 epch:18.29 loss:1.311 grdn:1.439 lr:4.9e-05 updt_s:0.364 data_s:0.771
INFO 2026-04-09 13:39:40 ot_train.py:451 step:3K smpl:27K ep:136 epch:19.43 loss:1.293 grdn:1.398 lr:4.4e-05 updt_s:0.366 data_s:0.750
INFO 2026-04-09 13:43:25 ot_train.py:451 step:4K smpl:29K ep:144 epch:20.57 loss:1.292 grdn:1.482 lr:3.9e-05 updt_s:0.366 data_s:0.757
INFO 2026-04-09 13:47:10 ot_train.py:451 step:4K smpl:30K ep:152 epch:21.71 loss:1.282 grdn:1.436 lr:3.4e-05 updt_s:0.364 data_s:0.760
INFO 2026-04-09 13:50:52 ot_train.py:451 step:4K smpl:32K ep:160 epch:22.86 loss:1.300 grdn:1.442 lr:2.9e-05 updt_s:0.368 data_s:0.740
INFO 2026-04-09 13:54:39 ot_train.py:451 step:4K smpl:34K ep:168 epch:24.00 loss:1.269 grdn:1.383 lr:2.5e-05 updt_s:0.365 data_s:0.765
INFO 2026-04-09 13:58:31 ot_train.py:451 step:4K smpl:35K ep:176 epch:25.14 loss:1.271 grdn:1.395 lr:2.1e-05 updt_s:0.366 data_s:0.792
INFO 2026-04-09 14:02:16 ot_train.py:451 step:5K smpl:37K ep:184 epch:26.29 loss:1.273 grdn:1.418 lr:1.7e-05 updt_s:0.366 data_s:0.760
INFO 2026-04-09 14:06:00 ot_train.py:451 step:5K smpl:38K ep:192 epch:27.43 loss:1.272 grdn:1.370 lr:1.3e-05 updt_s:0.368 data_s:0.749
INFO 2026-04-09 14:09:48 ot_train.py:451 step:5K smpl:40K ep:200 epch:28.57 loss:1.275 grdn:1.376 lr:1.0e-05 updt_s:0.368 data_s:0.769
INFO 2026-04-09 14:13:31 ot_train.py:451 step:5K smpl:42K ep:208 epch:29.71 loss:1.286 grdn:1.425 lr:7.8e-06 updt_s:0.368 data_s:0.747
INFO 2026-04-09 14:17:14 ot_train.py:451 step:5K smpl:43K ep:216 epch:30.86 loss:1.259 grdn:1.374 lr:5.8e-06 updt_s:0.366 data_s:0.748
INFO 2026-04-09 14:20:56 ot_train.py:451 step:6K smpl:45K ep:224 epch:32.00 loss:1.266 grdn:1.380 lr:4.2e-06 updt_s:0.367 data_s:0.742
INFO 2026-04-09 14:24:48 ot_train.py:451 step:6K smpl:46K ep:232 epch:33.14 loss:1.280 grdn:1.386 lr:3.1e-06 updt_s:0.366 data_s:0.789
INFO 2026-04-09 14:28:35 ot_train.py:451 step:6K smpl:48K ep:240 epch:34.29 loss:1.255 grdn:1.359 lr:2.6e-06 updt_s:0.365 data_s:0.770
INFO 2026-04-09 14:28:35 ot_train.py:471 Checkpoint policy after step 6000
INFO 2026-04-09 14:29:31 ot_train.py:545 End of training
```
StageC (VGG aux) skipped: path inactive for this job (variant=a, SMOLVLA_ENABLE_VGG=0).

- [PASS] stage loop driver executed
  - 2026-04-09T13:29:33Z
