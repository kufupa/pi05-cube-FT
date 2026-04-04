# phase09_vgg_gates
Executed: 2026-04-04T10:37:50Z

## SmolVLA velocity/value-gate validation
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
Loading  HuggingFaceTB/SmolVLM2-500M-Instruct weights ...
```

## Velocity field check
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
has VLAFlowMatching: True
VLAFlowMatching methods: ['forward', 'sample_actions']
```

- [WARN] Fallback velocity check used; full gate telemetry unavailable
  - 2026-04-04T10:39:04Z

- [WARN] VGG gate evaluation failed; StageC may be skipped
  - 2026-04-04T10:39:04Z
Gates not fully passed; stageC will rely on SMOLVLA_ENABLE_VGG and gate JSON.
