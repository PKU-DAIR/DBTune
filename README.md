# DBTune: Customized and Efficient Database Tuning System

**DBTune** is a customized and efficient database tuning system that can automatically find good configuration knobs for a database system. It support multiple tuning applications, including performance tuning, resource-oriented tuning or multiple-objective tuning defined by the users. DBTune is equipped with the start-of-the-art algorithms for tuning a database. It allows users to easily tune the database knobs without the pain of manually replaying the workloads and collecting the performance metrics of a database. DBTune is designed and developed by the database team from the <a href="https://cuibinpku.github.io/index.html" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University.





## Why DBTune?
- **Optimized for database tuning:** DBTune has customized functions and user-friendly interfaces for tuning the DBMSs. Users can conduct performance tuning, resource tuning or multiple-objective tuning via simply setting their tuning objectives.

- **Comprehensive technique supports and automatic selection:** DBTune is equipped with the state-of-the-art tuning techniques.
It automatically chooses the proper algorithms for a specific tuning task to separate algorithm selection complexity  away from the user.

- **Performance boosted by transfer learning:** DBTune extracts knowledge from historical tuning tasks to speed up the current tuning. The more users utilize DBTune, the faster the tuning would be. 

## Overview of Supported Techniques 
DBTune e supports the whole pipeline of configuration tuning, including knob selection, configuration tuning and knowledge transfer. 
Each module is equipped with multiple algorithm choices.
For a given tuning task, DBTune automatically selects a proper solution path among the choices.

<p align="left">
<img src="documents/image/techniques.jpg">
</p>

## Installation 
Installation Requirements:
- Python >= 3.6 

### Manual Installation from Source
To install the newest DBTune package, just type the following scripts on the command line:
 ```shell
   git clone git@github.com:Blairruc-pku/DBTuner.git && cd DBTune
   pip install -r requirements.txt
   pip install .
   ```




## Preparation 
####  Workload Preparation 
DBTune currently supports three database benchmarks:  <a href="https://github.com/oltpbenchmark/oltpbench.git" target="_blank" rel="nofollow">OLTP\-Bench</a>,  <a href="https://github.com/akopytov/sysbench.git" target="_blank" rel="nofollow">SYSBENCH</a>  and <a href="https://github.com/winkyao/join-order-benchmark" target="_blank" rel="nofollow">JOB</a>. 
Please reffer to the <a href="https://github.com/Blairruc-pku/DBTuner/blob/main/workload_prepare.md" target="_blank" rel="nofollow">details instuction</a>  for preparing the workloads.
####  Database Connection Setup
To provide the database connection information, the users need to edit the `config_auto.ini`.
```ini
db = mysql
host = 127.0.0.1
port = 3306
user = root
passwd =
  ```
DBTune currently supports to be deployed on MySQL and PostgreSQL using an integrated framework.
It provides several settings for database connections, including 
 <a href="https://github.com/Blairruc-pku/DBTuner/blob/main/documents/database_setting.md#remote--local-database" target="_blank" rel="nofollow"> remote/local database connection</a>,
<a href="https://github.com/Blairruc-pku/DBTuner/blob/main/documents/database_setting.md#tuning-non-dynamic-knobs" target="_blank" rel="nofollow">tuning non-dynamic knobs with restarts</a>, 
and <a href="https://github.com/Blairruc-pku/DBTuner/blob/main/documents/database_setting.md#tuning-with-resource-isolation" target="_blank" rel="nofollow">tuning with resource isolation</a>.
Please reffer to the <a href="https://github.com/Blairruc-pku/DBTuner/blob/main/documents/database_setting.md" target="_blank" rel="nofollow">details configurations</a>  for database connection.

## Quick Start

 
1. Specify the tuning objective.
Examples:
Performance tuning, e.g., maximizing throughputs.
```ini
task_id = performance1
performance_metric = ['tps']
```
Resource-oriented tuning, e.g., minimizing  cpu resource while throughputs > 200 txn/s and 95th percentile latency < 60 sec.

```ini
task_id = resource1
performance_metric = ['-cpu']
#constraints: Non-positive constraint values (”<=0”) imply feasibility.
constraints = ["200 - tps", "latency - 60"]
```

Multiple objective tuning, e.g., maximizing throughput and minimizing I/O.
```ini
task_id = mutiple1
performance_metric = ['tps', '-cpu]
reference_point = [0, 100]
```

2. Conduct Tuning.
```bash
python optimize.py  --config=config_performance.ini
```

For more information, please refer to the <a href="https://github.com/Blairruc-pku/DBTuner/blob/main/documents/tuning_setting.md#specific-tuning-setting" target="_blank" rel="nofollow">specific tuning settings </a>. 
