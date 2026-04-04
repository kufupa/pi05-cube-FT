# phase11_slurm_orchestration
Executed: 2026-04-04T12:44:22Z

## Workflow script check
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
usage: smolvla_workflow_launcher.py [-h] [--submit] [--write-json WRITE_JSON]

options:
  -h, --help            show this help message and exit
  --submit
  --write-json WRITE_JSON
```

## Submit helper check
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
workflow.json written: /vol/bitbucket/aa6622/pi05-cube-FT/runs/workflow_20260404_124422.json
workflow run id: 20260404_124422
workflow file: /vol/bitbucket/aa6622/pi05-cube-FT/runs/workflow_20260404_124422.json
```

## Watcher helper check
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
usage: watch_workflow.py [-h] --job-ids JOB_IDS [JOB_IDS ...] [--poll POLL]
                         [--max-retries MAX_RETRIES]

options:
  -h, --help            show this help message and exit
  --job-ids JOB_IDS [JOB_IDS ...]
  --poll POLL
  --max-retries MAX_RETRIES
```

- [PASS] orchestration helpers validated
  - 2026-04-04T12:44:22Z
