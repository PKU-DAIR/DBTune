from autotune.utils.config import parse_args
from autotune.database.mysqldb import MysqlDB
from autotune.database.postgresqldb import PostgresqlDB
from autotune.dbenv import DBEnv
from autotune.tuner import DBTuner
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.ini', help='config file')
    # [DBTune 2.0] Support CLI overrides for Generalist Training Loop
    parser.add_argument('--task_id', type=str, help='Override task_id from config (for shared history)')
    parser.add_argument('--max_runs', type=int, help='Override max_runs (for cycle length)')
    parser.add_argument('--fog_tier', type=str, help='Override fog_tier (for generalist training)')
    parser.add_argument('--global_t_base', type=int, help='Global step offset for JSONL logging (previous max_runs)')
    parser.add_argument('--steps_per_phase', type=int, help='Override steps_per_phase so alpha reaches 1.0 at end of this phase')
    parser.add_argument('--recalibration_interval', type=int, default=0, help='Recalibrate baseline TPS every N phase steps (0=disabled)')
    opt = parser.parse_args()


    args_db, args_tune = parse_args(opt.config)

    # [DBTune 2.0] Apply Overrides
    if opt.task_id:
        args_tune['task_id'] = opt.task_id
    if opt.max_runs:
        args_tune['max_runs'] = opt.max_runs
    if opt.fog_tier:
        args_tune['fog_tier'] = opt.fog_tier
    if opt.global_t_base is not None:
        args_tune['global_t_base'] = opt.global_t_base
    if opt.steps_per_phase is not None:
        args_tune['steps_per_phase'] = opt.steps_per_phase
    if opt.recalibration_interval:
        args_tune['recalibration_interval'] = opt.recalibration_interval

    import subprocess
    import time
    import sys

    def check_k8s_db_status(args_db):
        """Verifica que la BDD esté lista antes de empezar"""
        if args_db.get('deployment_mode') != 'kubernetes':
            return True
            
        ns = args_db.get('k8s_pg_namespace', 'pg')
        cluster = args_db.get('k8s_pg_cluster_name', 'pg-cluster')
        
        print(f"🏥 Realizando Pre-Flight Check de la Base de Datos ({cluster} en {ns})...")
        
        try:
            # 1. Buscar Pod Primario
            cmd_find = f"kubectl get pod -n {ns} -l cnpg.io/cluster={cluster},cnpg.io/instanceRole=primary -o jsonpath='{{.items[0].metadata.name}}'"
            result = subprocess.run(cmd_find, shell=True, capture_output=True, text=True)
            primary_pod = result.stdout.strip()
            
            if not primary_pod:
                print(f"❌ No se encontró ningún Pod Primario activo para el cluster {cluster}")
                return False
                
            print(f"ℹ️  Pod Primario detectado: {primary_pod}")
            
            # 2. Verificar pg_isready
            cmd_check = f"kubectl exec -n {ns} {primary_pod} -- pg_isready -U postgres"
            result_check = subprocess.run(cmd_check, shell=True, capture_output=True, text=True)
            
            if result_check.returncode == 0 and "accepting connections" in result_check.stdout:
                print("✅ PostgreSQL está HEALTHY y aceptando conexiones. Iniciando experimento...")
                return True
            else:
                print(f"❌ PostgreSQL no está listo: {result_check.stdout} {result_check.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error durante el chequeo de la BDD: {e}")
            return False

    if args_db['db'] == 'mysql':
        db = MysqlDB(args_db)
    elif args_db['db'] == 'postgresql':
        db = PostgresqlDB(args_db)

    # CHECK ANTES DE INICIAR
    if not check_k8s_db_status(args_db):
        print("🚨 ABORTANDO: La base de datos no está lista. Por favor, revisa el cluster.")
        sys.exit(1)

    # Export task_id so dbenv.py can name the training_steps JSONL correctly.
    import os as _os
    _os.environ['TASK_ID'] = args_tune.get('task_id', 'unknown')

    try:
        # Initialize Environment and Tuner (Moved inside try/except to catch startup errors)
        env = DBEnv(args_db, args_tune, db)
        tuner = DBTuner(args_db, args_tune, env)
        tuner.tune()
    except Exception as e:
        import traceback
        import sys
        print("="*60)
        print("🚨 CRITICAL ERROR: DBTune Crashed")
        print(f"Error: {e}")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)

