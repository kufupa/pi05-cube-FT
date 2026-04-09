# phase04_metaworld_install
Executed: 2026-04-09T00:15:48Z

## Meta-World install (pip)
```
Requirement already satisfied: metaworld>=0.1.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (3.0.0)
Requirement already satisfied: gymnasium>=1.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from metaworld>=0.1.0) (1.2.3)
Requirement already satisfied: mujoco>=3.0.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from metaworld>=0.1.0) (3.6.0)
Requirement already satisfied: numpy>=1.18 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from metaworld>=0.1.0) (2.2.6)
Requirement already satisfied: scipy>=1.4.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from metaworld>=0.1.0) (1.17.1)
Requirement already satisfied: imageio in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from metaworld>=0.1.0) (2.37.3)
Requirement already satisfied: cloudpickle>=1.2.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gymnasium>=1.1->metaworld>=0.1.0) (3.1.2)
Requirement already satisfied: typing-extensions>=4.3.0 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gymnasium>=1.1->metaworld>=0.1.0) (4.15.0)
Requirement already satisfied: farama-notifications>=0.0.1 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from gymnasium>=1.1->metaworld>=0.1.0) (0.0.4)
Requirement already satisfied: absl-py in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from mujoco>=3.0.0->metaworld>=0.1.0) (2.4.0)
Requirement already satisfied: etils[epath] in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from mujoco>=3.0.0->metaworld>=0.1.0) (1.14.0)
Requirement already satisfied: glfw in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from mujoco>=3.0.0->metaworld>=0.1.0) (2.10.0)
Requirement already satisfied: pyopengl in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from mujoco>=3.0.0->metaworld>=0.1.0) (3.1.10)
Requirement already satisfied: fsspec in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from etils[epath]->mujoco>=3.0.0->metaworld>=0.1.0) (2026.2.0)
Requirement already satisfied: zipp in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from etils[epath]->mujoco>=3.0.0->metaworld>=0.1.0) (3.23.0)
Requirement already satisfied: pillow>=8.3.2 in /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages (from imageio->metaworld>=0.1.0) (11.3.0)
```

## push-v3 env probe
```
env class: SawyerPushEnvV3
has reset: True
```

- [PASS] Meta-World install + push-v3 reset smoke passed
  - 2026-04-09T00:16:07Z
