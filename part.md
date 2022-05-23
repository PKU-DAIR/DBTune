### Remote / Local Database
The database to tune can be on the local host, or on a remote host.
Please take care to set the values of the following parameters in `config.ini`.
```ini
remote_mode = True
ssh_user = 

db = mysql
host = 127.0.0.1
port = 3306
user = root
passwd =
sock = ~/mysql/base/mysql.sock
cnf = ~/cnf/mysql.cnf
```
- If you are tuning a database on the **local** host: 
  Please set `remote_mode=False`, 
  along with the followed database related parameters.

- If you are tuning a database on a **remote** host: 
  Please set `remote_mode=True`, 
and set `ssh_user` to your username for login to the remote host. 
  The followed parameters should be set according to the information of the remote database.
  
    - Note, before remote mode tuning, you should also set up **SSH Passwordless Login**. Specifically,
        1. Create public and private keys using ssh-key-gen on local-host
           
           ```ssh-keygen-t rsa```
           
            which will generate `id-rsa` and `id-rsa.pub` in the `~/.ssh` directory, 
           where `id-rsa.pub` is the public key.
        2. Copy the public key to `~/.ssh/authorized_keys` in remote-host
        3. Try connecting the remote host without password by `ssh -l SSH_USERNAME HOST`
    

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

### Others

#### Continuous Tuning 
There might be some cases where the tuning task is interrupted, 
or where users find finished tuning tasks have not converged and want to train more step.

In these cases, user can optionally choose to continuously tune a former task, by setting the same `task_id`. 
However, please check that other tuning parameters are the same as the former task (e.g. `performance_metric`, `constraints`, etc.)

