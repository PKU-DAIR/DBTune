from autotune.utils.config import parse_args
from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv

if __name__ == '__main__':
    args_db, args_tune = parse_args()

    if args_db['db'] == 'mysql':
        db = MysqlDB(args_db)
    elif args_db['db'] == 'postgresql':
        db = PostgresqlDB(args_db)


    env = DBEnv(args_db, args_tune, db)

    # MySQL - OFFLINE MODE
    # DBEnv to test:
    # 1. get_states (OK)
    # 2. initialize, step_GP

    # --> MysqlDB to test: (OK)
    # 1. apply_knobs_online (OK)
    # 2. apply_knobs_offline --> db: kill, start, gen_config (OK)
    #   1) kill (OK)
    #   2) gen_config (OK)
    #   3) start (OK)

    # --> PostgresqlDB to test: (OK)
    # 1. apply_knobs_online (OK)
    # 2. apply_knobs_offline --> db: kill, start, gen_config (OK)
    #   1) kill (OK)
    #   2) gen_config (OK)
    #   3) start (OK)

    # To test
    print(env.step_GP(db.default_knobs))

    # print(env.get_states(collect_resource=True))
    # db.apply_knobs_online(db.default_knobs)
    # db._kill_mysqld()
    # print(db._start_mysqld())
    # db._gen_config_file(db.default_knobs)
    # db.apply_knobs_offline(db.default_knobs)
    # env.initialize(collect_CPU=True)


