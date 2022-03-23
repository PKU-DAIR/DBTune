from autotune.utils.config import parse_args, parse_knob_config
import pdb
from autotune.database import MysqlDB, PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.tuner import DBTuner


if __name__ == '__main__':
    args_db, args_tune = parse_args()

    if args_db['db'] == 'mysql':
        db = MysqlDB(args_db)
    elif args_db['db'] == 'postgresql':
        db = PostgresqlDB(args_db)

    env = DBEnv(args_db, args_tune, db)
    tuner = DBTuner(args_db, args_tune, env)
    tuner.tune()

