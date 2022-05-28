import json
import sys
from autotune.utils.config import parse_args
from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.tuner import DBTuner
import argparse

def parse(run_history, apply_option):
    f = open(run_history)
    res = json.load(f)['data']
    if apply_option == 'default':
        config = res[0]["configuration"]
    else:
        config, best_tps = res[0], 0
        for c in res:
            if "tps"  in c["external_metrics"].keys() and  c["external_metrics"]["tps"] > best_tps:
                config = c["configuration"]
                best_tps = c["external_metrics"]["tps"]
    return config

if __name__ == '__main__':
    run_history = sys.argv[1]
    apply_option = sys.argv[2]
    config = parse(run_history, apply_option)

    args_db, args_tune = parse_args("/root/demo/DBTuner-main/scripts/config.ini")
    if args_db['db'] == 'mysql':
        db = MysqlDB(args_db)
    elif args_db['db'] == 'postgresql':
        db = PostgresqlDB(args_db)

    env = DBEnv(args_db, args_tune, db)
    timeout, metrics, internal_metrics, resource = env.step_GP(config)
    print(' objective value: %s.' % (metrics[0]))








