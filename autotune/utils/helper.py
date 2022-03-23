import os
import sys
from dynaconf import settings

def check_env_setting(workload_name, rds_mode=False):
    from ..knobs import logger
    from .autotune_exceptions import AutotuneError
    env_vars = ['MYSQLD', 'MYCNF', 'MYSQL_SOCK']
    if rds_mode:
        env_vars = []
    for env_var in env_vars:
        # if not os.environ.get(env_var) and not settings[env_var]:
        if not os.environ.get(env_var):
            err_msg = 'You should set {} env var before train'.format(env_var)
            logger.error(err_msg)
            raise AutotuneError(err_msg)
    if workload_name == 'sysbench':
        # if not os.environ.get('SYSBENCH_BIN') and not settings['SYSBENCH_BIN']:
        if not os.environ.get('SYSBENCH_BIN'):
            err_msg = 'You should set SYSBENCH_BIN before train'
            logger.error(err_msg)
            raise AutotuneError(err_msg)
    elif workload_name == 'oltpbench':
        # if not os.environ.get('OLTPBENCH_BIN') and not settings['OLTPBENCH_BIN']:
        if not os.environ.get('OLTPBENCH_BIN'):
            err_msg = 'You should set OLTPBENCH_BIN before train'
            logger.error(err_msg)
            raise AutotuneError(err_msg)

def gen_env_vars(argv):
    env_vars = ['MYSQLD',
                'MYCNF',
                'MYSQL_SOCK',
                'TPCC_HOME',
                'SYSBENCH_BIN',
                'OLTPBENCH_BIN',
                'MYSQL_PORT',
                ]
    with open(argv[1], 'w') as fout:
        fout.write("#! /bin/bash\n")
        for var in env_vars:
            if settings[var] is not None:
                # if not os.path.exists(settings[var]):
                #     err_msg = "ERROR: ENVIRONMENT Var: \"{}\" : \"{}\" not exist!".format(var, settings[var])
                #     print(err_msg)
                #     return
                cmd = "export {}=\"{}\"\n".format(var, settings[var])
                fout.write(cmd)
            else:
                err_msg = "ERROR: \"{}\" is missing in setting file".format(var)
                print(err_msg)
                return
        print("Env file is generated, plz run \"source {} \" before autotune".format(argv[1]))

if __name__ == '__main__':
    gen_env_vars(sys.argv)
