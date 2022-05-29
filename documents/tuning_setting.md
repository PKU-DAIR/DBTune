# Tuning Setting
DBTune provides automatic algorithm selection and knowledge transfer modules so that the users do not need to disturb themselves with choosing the proper algorithm to solve a specific problem.
To conduct a specific tuning setting, the user could customize the algorithms for  knobs selection, configuration optimization and knowledge transfer.

## Automatic Algorithm Selection

DBTune currently supports 9 performance metrics, including tps, lat, qps, cpu, IO, readIO, writeIO, virtualMem, physical.
To conduct tuning, the users only needs to setup their optimization objectives in `config_auto.ini`. 

To conduct performance tuning, e.g., maximizing throughputs.
```ini
task_id = performance1
performance_metric = ['tps']
```
To conduct resource-oriented tuning, e.g., minimizing  cpu resource while throughputs > 200 txn/s and 95th percentile latency < 60 sec.

```ini
task_id = resource1
performance_metric = ['-cpu']
#constraints: Non-positive constraint values (”<=0”) imply feasibility.
constraints = ["200 - tps", "latency - 60"]
```

DBTune also support multiple objective tuning, e.g., maximizing throughput and minimizing I/O.
```ini
task_id = mutiple1
performance_metric = ['tps', '-cpu]
reference_point = [0, 100]
```

## Continuous Tuning and Automatic Knowledge Transfer
Each tuning task is identified via its task id. 
Using the same task_id leads to tuning based the previous tuning data. Therefore, set a different task_id when setup a new tuning task.
#### Continuous Tuning
There might be some cases where the tuning task is interrupted, or where users find finished tuning tasks have not converged and want to train more step.
In these cases, user can optionally choose to continuously tune a former task, by setting the same `task_id`. 
However, please check that other tuning parameters are the same as the former task (e.g. `performance_metric`, `constraints`, etc.)

#### Automatic Knowledge Transfer
The tuning history for each tuning task is stored in the directory `DBTune_history` by default.
Everytime the users starts a new tuning task, the historical knowledge is extracted to speed up the target tuning.

 


## Specific Tuning Setting
`config.ini` provides all the tuning settings for DBTune and their corresponding explanation.
The users could customize their tuning algorithms for knobs selection, configuration optimization and knowledge transfer.

### Knob Selection Algorithms
We have implemented 5 knob selection algorithms that users can choose to use, 
namely SHAP, fANOVA, LASSO, Gini Score, and Ablation Analysis. All related parameters are listed as follows:

```ini
knob_config_file =  ~/DBTuner/scripts/experiment/gen_knobs/JOB_shap.json
knob_num = 20

selector_type = shap

initial_tunable_knob_num = 20
incremental = increase
incremental_every = 10
incremental_num = 2
```

Specifically, the algorithms are used as knob importance measures in 2 stages.

- **Stage One: Determine initial knob configuration space.**
  
    there are two related parameters in `config.ini`
  
    1. `knob_config_file`: 
       Users can either (1) choose one of the knob configuration files we provided in `./scripts/experiment/gen_knobs`, 
       where knobs are ranked by analyzing 600+ samples of workload SYSBENCH or JOB,
       or (2) provide their knob configuration file according to their specific tuning tasks. 
    2. `knob_num`: (1) If users want to tune all the knobs in `knob_config_file`, please set `knob_num` to the number of knobs it contains.
       (2) If users only want to tune part of the knobs considering tuning efficiency, they can set the parameter to a smaller number, and further set:
       - `initial_runs`, which indicates the number of sampling steps for knob importance analysis;
       - `selector_type`, which indicates the selection method used for important knobs selection.
    

- **Stage Two: Whether to tune knobs incrementally.**

    After determining the initial knob configuration space, there are 3 ways of further tuning by setting the parameter`incremental` to:
    1. `none`, which indicates none-incremental tuning, and the knob configuration space will be fixed.
    2. `increase`: According to OtterTune, the number of tuning knobs will increase from a smaller one to a bigger one, by setting
        `incremental_every` and `incremental_num`.
    3. `decrease`: According to Tuneful, the number of tuning knobs will decrease from a bigger one to a smaller one.

**Note** that knob selection only supports single-objective optimization tasks, 
and that the incremental knob tuning only supports MBO and SMAC optimization algorithms.







### Configuration Optimization Algorithms 

DBTune currently supports 6 configuration optimizers that users can choose to use, namely MBO, SMAC, TPE, DDPG, TurBO and GA.
```ini
############Optimizer###############
# tuning method [MBO, SMAC, TPE, DDPG, TurBO, GA]
optimize_method = SMAC
```
Here are some optimizer-specific setting.
```ini
###TurBO####
# whether TurBO start from the scratch
tr_init = True

###DDPG####
batch_size = 16
mean_var_file = mean_var_file.pkl
# dir for memory pool
replay_memory =
# dir for params
params = model_params/11111_135
```

### Knowledge Transfer
Related parameters are listed as follows:
```ini
############Transfer###############
# transfer_framework :[none, workload_map, rgpe, finetune]
transfer_framework = rgpe
# dir of source data for mapping
data_repo = /logs/bo_history
```

DBTune support 3 transfer frameworks:  workload_map, rgpe, finetune.
It uses the data in `/logs/bo_history`  as source data for tansfer.
Turn off Knowledge Transfer by setting "transfer_framework" = none.

## Plot Convergence
DBTune supports visualization for single-objective optimization tasks.
After the optimization, convergence plot will be save to `task_id.png`



