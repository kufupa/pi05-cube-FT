# phase07_jepa_setup
Task: push-v3
Executed: 2026-04-04T00:02:11Z

## JEPA smoke unroll
```
[jepa-smoke] torch 2.6.0+cu124
[WARNING ][2026-04-04 01:02:45][Cluster utils       ][_build_dataset_paths     ] JEPAWM_DSET environment variable not set. Dataset paths will need to be provided manually or set the environment variable. See README.md for setup instructions.
[INFO    ][2026-04-04 01:02:45][src.utils.yaml_utils][replace_var              ] Expanded environment variable 'JEPAWM_LOGS' to '/vol/bitbucket/aa6622/.cache/jepa_workflow' at folder
[INFO    ][2026-04-04 01:02:45][src.utils.yaml_utils][replace_var              ] Expanded environment variable 'JEPAWM_CKPT' to '/vol/bitbucket/aa6622/.cache/jepa_workflow' at checkpoint_folder
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ] Using hardcoded dimensions for metaworld: action_dim=4, proprio_dim=4
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ] Preprocessor attributes for metaworld:
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ]   action_mean: tensor([ 0.0057,  0.1574, -0.1396,  0.1998])
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ]   action_std: tensor([0.7359, 0.7341, 0.7183, 0.7432])
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ]   proprio_mean: tensor([-0.0029,  0.6544,  0.1541,  0.6402])
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ]   proprio_std: tensor([0.1043, 0.1123, 0.0798, 0.2614])
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ]   state_mean: tensor([-0.0029,  0.6544,  0.1541,  0.6402,  0.0081,  0.6767,  0.0943,  0.3860,
        -0.0506,  0.0343,  0.4751,  0.0179,  0.0441,  0.0089,  0.0238,  0.0000,
         0.0000,  0.0000, -0.0030,  0.6530,  0.1545,  0.6445,  0.0081,  0.6762,
         0.0943,  0.3860, -0.0507,  0.0343,  0.4754,  0.0178,  0.0441,  0.0089,
         0.0238,  0.0000,  0.0000,  0.0000,  0.0157,  0.7268,  0.1240])
[INFO    ][2026-04-04 01:02:45][root                ][_load_model_with_config  ]   state_std: tensor([0.1043, 0.1123, 0.0798, 0.2614, 0.1051, 0.1019, 0.0881, 0.4468, 0.1840,
        0.1491, 0.4722, 0.0669, 0.1602, 0.0322, 0.1525, 0.0000, 0.0000, 0.0000,
        0.1038, 0.1124, 0.0795, 0.2626, 0.1049, 0.1017, 0.0881, 0.4467, 0.1838,
        0.1487, 0.4722, 0.0666, 0.1602, 0.0322, 0.1525, 0.0000, 0.0000, 0.0000,
        0.1418, 0.1211, 0.0963])
[INFO    ][2026-04-04 01:02:49][dinov2              ][__init__                 ] using MLP layer as FFN
[INFO    ][2026-04-04 01:02:50][root                ][init_video_model         ] Encoder: DinoEncoder(
  (base_model): DinoVisionTransformer(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
      (norm): Identity()
    )
    (blocks): ModuleList(
      (0-11): 12 x NestedTensorBlock(
        (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        (attn): MemEffAttention(
          (qkv): Linear(in_features=384, out_features=1152, bias=True)
          (proj): Linear(in_features=384, out_features=384, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
    )
    (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
    (head): Identity()
  )
)
[INFO    ][2026-04-04 01:02:50][root                ][init_video_model         ] 🧠 Encoder: DinoEncoder (22,056,576 params, frozen=True)
[INFO    ][2026-04-04 01:02:50][app.plan_common.models.AdaLN_vit][__init__                 ] 🔧 Initialized 6 AdaLN blocks (scale_factor=10)
[INFO    ][2026-04-04 01:02:50][root                ][init_video_model         ] Predictor: VisionTransformerAdaLN(
  (predictor_embed): Linear(in_features=384, out_features=384, bias=True)
  (action_encoder): Linear(in_features=20, out_features=400, bias=True)
  (predictor_blocks): ModuleList(
    (0-5): 6 x FWAdaLNBlock(
      (norm1): LayerNorm((400,), eps=1e-06, elementwise_affine=True)
      (attn): RoPEAttention(
        (qkv): Linear(in_features=400, out_features=1200, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=400, out_features=400, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): Identity()
      (norm2): LayerNorm((400,), eps=1e-06, elementwise_affine=True)
      (mlp): MLP(
        (fc1): Linear(in_features=400, out_features=1600, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=1600, out_features=400, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
      )
      (adaLN_modulation): Sequential(
        (0): SiLU()
        (1): Linear(in_features=400, out_features=2400, bias=True)
      )
    )
  )
  (predictor_norm): LayerNorm((400,), eps=1e-06, elementwise_affine=True)
  (predictor_proj): Linear(in_features=384, out_features=384, bias=True)
)
[INFO    ][2026-04-04 01:02:50][root                ][init_video_model         ] 🔮 Predictor: VisionTransformerAdaLN (17,630,480 params)
[INFO    ][2026-04-04 01:02:54][root                ][fetch_checkpoint         ] Loading checkpoint from local path: /vol/bitbucket/aa6622/.cache/jepa_workflow/mw_final_sweep/mw_4f_fsk5_ask1_r224_pred_AdaLN_ftprop_depth6_repro_2roll_save/seed100k/jepa-latest.pth.tar
[INFO    ][2026-04-04 01:02:55][root                ][load_checkpoint_state_dict] loaded pretrained predictor from epoch 50 with msg: <All keys matched successfully>
[INFO    ][2026-04-04 01:02:55][root                ][load_checkpoint_state_dict] loaded pretrained proprio encoder from epoch 50 with msg: <All keys matched successfully>
[INFO    ][2026-04-04 01:02:55][root                ][load_checkpoint          ] Downloading checkpoint from URL: https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar
[INFO    ][2026-04-04 01:05:17][root                ][load_checkpoint          ] loaded pretrained trainable module from epoch 29 with msg: _IncompatibleKeys(missing_keys=['decoder_pos_embed'], unexpected_keys=[])
[INFO    ][2026-04-04 01:05:18][root                ][init_module              ] loaded pretrained head named image_head
[INFO    ][2026-04-04 01:05:18][root                ][__init__                 ] 📉 Loss weights: L2=1.0
[INFO    ][2026-04-04 01:05:18][root                ][_load_model_with_config  ] Loaded encoder and predictor
[jepa-smoke] model loaded
{
  "repo": "/vol/bitbucket/aa6622/VGG JEPA/jepa-wms",
  "task": "push-v3",
  "smoke_steps": 8,
  "status": "fail",
  "errors": [
    "shape '[1, -1, 20]' is invalid for input of size 32"
  ]
}

## Root-cause debugging notes (2026-04-04)
- `jepa_smoke_check.py` successfully loads the repo config and model weights for `push-v3`, including `action_encoder_inpred=true` and predictor stack construction.
- Failure pattern when validating rollout: raw action suffix with shape `[8,1,4]` produces `shape '[1, -1, 20]' is invalid for input of size 32`.
- `model.unroll` with scaled action suffix `[8,1,20]` shifts failure to `Given normalized_shape=[400], expected input with shape [*, 400], but got input of size[1, 512, 384]`.
- Interpretation:
  - This indicates inconsistent predictor contracts in the selected JEPA variant (visual token dimension 384 vs action-path/normalization expecting 400 in the same rollout graph), not an environment or module-availability issue.
  - GPU-only smoke on `gpucluster3` is not feasible for deeper probing (previous full-path runs were terminated by host resource limits), so this was reproduced via lightweight smoke probes and log inspection only.
- `phase07` is therefore marked as **blocked by upstream JEPA model-shape mismatch**, while load/setup milestones remain complete.
