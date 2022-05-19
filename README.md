# DBTune: Customized and Efficient Database Tuning System

**DBTune** is a customized and efficient database tuning system that can automatically find good configuration knobs for a database system. It support multiple tuning applications, including performance tuning, resource-oriented tuning or multiple-objective tuning defined by the users. DBTune is equipped with the start-of-the-art algorithms for tuning a database. It allows users to easily tune the database knobs without the pain of manually replaying the workloads and collecting the performance metrics of a database. DBTune is designed and developed by the database team from the <a href="https://cuibinpku.github.io/index.html" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University.





## Why DBTune?
- **Optimized for database tuning:** DBTune has customed functions and user-friendly interfaces for tuning databases. Users can conduct performance tuning, resource tuning or multiple-objective tuning for DBMSs via a few lines of codes.

- **Comprehensive tuning supports and automatic selection:** DBTune supports the whole pipeline of configuration tuning, including knob selection, configuration tuning and knowledge transfer. Each module is equipped with multiple algorithm choices.

- **Ease of Deployment:** DBTune is easy to deploy on a variety of DBMSs using an integrated framework.

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




## Workload Preparation 

DBTune currently supports three database benchmarks:  <a href="https://github.com/oltpbenchmark/oltpbench.git" target="_blank" rel="nofollow">OLTP\-Bench</a>,  <a href="https://github.com/akopytov/sysbench.git" target="_blank" rel="nofollow">SYSBENCH</a>  and <a href="https://github.com/winkyao/join-order-benchmark" target="_blank" rel="nofollow">JOB</a>. Please reffer to the <a href="https://github.com/Blairruc-pku/DBTuner/blob/main/workload_prepare.md" target="_blank" rel="nofollow">details instuction</a>  for preparing the workloads.

## Quick Start

### Performance Tuning
1. Edit the config_performance.ini in the script.
 ```
 
  ```
3. 
