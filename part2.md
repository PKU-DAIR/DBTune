### Tuning non-dynamic knobs
Tuning non-dynamic knobs needs restarting the database.
When tuning  non-dynamic knobs, please set the "online_mode" False in `config.ini`.
```ini
####### Online tuning related
# whether not restart db
online_mode = False
```
### Tuning with resource isolation
DBTune provides a function to restrict the resource usage of the database instance via Cgroup.
Please modify the following parameters in `config.ini`.
```ini
####### Online tuning related
####### Resource isolation related
# whether isolate resource usage
isolation_mode = True
# pid for resource isolation in online tuning
pid = 4110
online_mode = False
```
And  setup a cgroup named server with its resource maximum usage specified.

```bash
sudo    cgcreate -g memory:server;
sudo   cgset -r memory.limit_in_bytes='16G' server
sudo  cgget -r memory.limit_in_bytes server
cgclassify -g memory:server  122220
sudo  cgcreate -g cpuset:server
sudo  cgset -r cpuset.cpus=0-7 server
```



### Performance tuning


#### Tuning Objective
Related parameters are listed as follows:
```ini
# task id
task_id = test8
# performance_metric: [tps, lat, qps, cpu, IO, readIO, writeIO, virtualMem, physical]
# default maximization, '- 'minus means minimization
performance_metric = ['-cpu']
# set for multi-objective tuning
reference_point = [None, None]
#constraints: Non-positive constraint values (”<=0”) imply feasibility.
constraints = ["100-tps", "readIO - 100"]
```
Each tuning task is identified via its task id. Using the same task_id leads to tuning based the previous tuning data. Therefore, set a different task_id when setup a new tuning task.
DBTune supports 9 tuning metrics. 
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
performance_metric = ['tps', '-IO']
```

#### Tuning Optimizers 

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




