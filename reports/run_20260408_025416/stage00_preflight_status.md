# phase00_inventory
Executed: 2026-04-09T00:09:47Z
## Host / runtime
ROOT=/vol/bitbucket/aa6622
USER=aa6622
HOST=merlin.doc.ic.ac.uk
Linux merlin.doc.ic.ac.uk 6.17.0-19-generic #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar  6 23:08:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux

## Scheduler probes
```
PARTITION   AVAIL  TIMELIMIT  NODES  STATE NODELIST
docker         up 3-00:00:00      2    mix gpuvm[25,27]
docker         up 3-00:00:00      1   idle kingfisher
a40            up 3-00:00:00      1   mix- gpuvm33
a40            up 3-00:00:00      1  down* hopper
a40            up 3-00:00:00      1   idle gpuvm34
a30            up 3-00:00:00      2    mix clapper,dipper
t4             up 3-00:00:00      9   idle gpuvm[13-19,24],kingfisher
a16*           up 3-00:00:00      3   idle gpuvm[35-36],parrot
a100           up 3-00:00:00      1   mix- linnet
a100           up 3-00:00:00      2    mix merlin,vm-he-a100
interactive    up 3-00:00:00      8   idle gpuvm[13-19,24]
training       up 3-00:00:00      2    mix gpuvm[25,27]
Configuration data as of 2026-04-09T01:09:47
AccountingStorageBackupHost = (null)
AccountingStorageEnforce = associations,limits,qos
AccountingStorageHost   = localhost
AccountingStorageExternalHost = (null)
AccountingStorageParameters = (null)
AccountingStoragePort   = 6819
AccountingStorageTRES   = cpu,mem,energy,node,billing,fs/disk,vmem,pages,gres/gpu,gres/gpumem,gres/gpuutil,gres/shard
AccountingStorageType   = accounting_storage/slurmdbd
AccountingStorageUser   = N/A
AccountingStoreFlags    = (null)
AcctGatherEnergyType    = (null)
AcctGatherFilesystemType = (null)
AcctGatherInterconnectType = (null)
AcctGatherNodeFreq      = 0 sec
AcctGatherProfileType   = (null)
AllowSpecResourcesUsage = no
AuthAltTypes            = auth/jwt
AuthAltParameters       = jwt_key=/etc/slurm/jwt_hs256.key
AuthInfo                = (null)
AuthType                = auth/munge
BatchStartTimeout       = 10 sec
BcastExclude            = /lib,/usr/lib,/lib64,/usr/lib64
BcastParameters         = (null)
BOOT_TIME               = 2026-03-20T15:31:20
BurstBufferType         = (null)
CertmgrParameters       = (null)
CertmgrType             = (null)
CliFilterPlugins        = (null)
ClusterName             = compute-cluster
CommunicationParameters = (null)
CompleteWait            = 0 sec
CpuFreqDef              = Unknown
CpuFreqGovernors        = OnDemand,Performance,UserSpace
CredType                = cred/munge
DataParserParameters    = (null)
DebugFlags              = CPU_Bind,Gres,JobComp,Power
DefMemPerNode           = UNLIMITED
DependencyParameters    = (null)
DisableRootJobs         = no
EioTimeout              = 60
EnforcePartLimits       = ANY
Epilog[0]               = /etc/slurm/epilog.d/*.sh
EpilogMsgTime           = 2000 usec
EpilogSlurmctld[0]      = /etc/slurm/slurmctldepilog.sh
FairShareDampeningFactor = 1
FederationParameters    = (null)
FirstJobId              = 1
GetEnvTimeout           = 120 sec
GresTypes               = gpu,shard
GpuFreqDef              = (null)
GroupUpdateForce        = 1
GroupUpdateTime         = 600 sec
HASH_VAL                = Match
HashPlugin              = hash/k12
HealthCheckInterval     = 0 sec
HealthCheckNodeState    = ANY
HealthCheckProgram      = (null)
InactiveLimit           = 0 sec
InteractiveStepOptions  = --interactive --preserve-env --pty $SHELL
JobAcctGatherFrequency  = 30
JobAcctGatherType       = jobacct_gather/cgroup
JobAcctGatherParams     = (null)
JobCompHost             = localhost
JobCompLoc              = http://cloud-vm-45-09.doc.ic.ac.uk:9200/slurmdev/_doc
JobCompParams           = (null)
JobCompPort             = 0
JobCompType             = jobcomp/elasticsearch
JobCompUser             = root
JobContainerType        = (null)
JobDefaults             = (null)
JobFileAppend           = 0
JobRequeue              = 1
JobSubmitPlugins        = lua
KillOnBadExit           = 0
KillWait                = 30 sec
LaunchParameters        = use_interactive_step,ulimit_pam_adopt,slurmstepd_memlock
Licenses                = (null)
LogTimeFormat           = iso8601_ms
MailDomain              = (null)
```

## GPU and CUDA probes
```
Thu Apr  9 01:09:47 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          On  |   00000000:02:00.0 Off |                    0 |
| N/A   27C    P0             45W /  300W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
torch: PASS
numpy: PASS
gymnasium: PASS
mujoco: PASS
```

## Offline/IO probes
```
ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers
pwd /vol/bitbucket/aa6622/pi05-cube-FT
```

- [PASS] preflight report saved to /vol/bitbucket/aa6622/pi05-cube-FT/reports/run_20260408_025416/stage00_preflight_status.md
  - 2026-04-09T00:09:58Z
