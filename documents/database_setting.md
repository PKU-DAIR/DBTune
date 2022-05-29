#  Tuning Setup
DBTune support tuning remote/local database, tuning non-dynamic knobs with restart, and tuning with resource isolation.
All you need is to configure the setting in `config_auto.ini`.

## Remote / Local Database
The database to tune can be on the local host, or on a remote host.
Please take care to set the values of the following parameters in `config_auto.ini`.
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
  Other parameters should be set according to the information of the remote database.
  
    - Note, before remote mode tuning, you should also set up **SSH Passwordless Login**. Specifically,
        1. Create public and private keys using ssh-key-gen on local-host
           
           ```ssh-keygen-t rsa```
           
            which will generate `id-rsa` and `id-rsa.pub` in the `~/.ssh` directory, 
           where `id-rsa.pub` is the public key.
        2. Copy the public key to `~/.ssh/authorized_keys` in remote-host
        3. Try connecting the remote host without password by `ssh -l SSH_USERNAME HOST`
    - A remote resource monitor process should be started on remote host before tuning, by executing:
        ```python
        python script/remote_resource_monitor.py
        ```

## Tuning non-dynamic knobs
Tuning non-dynamic knobs needs restarting the database.
When tuning  non-dynamic knobs, please set the "online_mode" False in `config_auto.ini`.
```ini
####### Online tuning related
# whether not restart db
online_mode = False
```
## Tuning with resource isolation
DBTune provides a function to restrict the resource usage of the database instance via Cgroup.
Please modify the following parameters in `config_auto.ini`.
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


