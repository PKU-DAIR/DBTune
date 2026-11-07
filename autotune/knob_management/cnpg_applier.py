"""
Aplicador de knobs para CloudNativePG
"""
import json
import subprocess
from typing import Dict
from autotune.knob_management.pg_knob_classifier import PgKnobClassifier

import logging
logger = logging.getLogger(__name__)

class CnpgKnobApplier:
    # Knobs que necesitan conversión de unidades
    BLOCK_SIZE_KB = 8  # PostgreSQL usa bloques de 8KB

    # Knobs que vienen en bloques (8KB) y necesitan MB
    BLOCK_KNOBS = {'shared_buffers', 'effective_cache_size', 'wal_buffers'}

    # Knobs que vienen en KB — mantener precisión kB (no truncar a MB)
    KB_KNOBS = {'work_mem', 'maintenance_work_mem'}

    # Knobs que ya son MB (solo añadir sufijo)
    MB_KNOBS: set = set()

    # Knobs que son segundos (añadir sufijo 's')
    SECONDS_KNOBS = {'checkpoint_timeout'}

    # Knobs que son milisegundos (añadir sufijo 'ms')
    MS_KNOBS = {'wal_writer_delay', 'bgwriter_delay'}

    # Knobs que son enteros sin unidad
    INTEGER_KNOBS = {'bgwriter_lru_maxpages', 'bgwriter_flush_after',
                     'max_parallel_workers_per_gather', 'max_worker_processes'}
    
    def _get_current_pod_uid(self):
        """Obtiene el UID del pod actual para detectar reinicios"""
        try:
            cmd = [
                'kubectl', 'get', 'pod',
                '-n', self.namespace,
                '-l', f'cnpg.io/cluster={self.cluster_name}',
                '--field-selector=status.phase=Running',
                '-o', 'jsonpath={.items[0].metadata.uid}'
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"No se pudo obtener Pod UID: {e}")
            return None

    def _wait_for_postgres_restart(self, old_uid, timeout_seconds=600):
        """Calcula el reinicio detectando cambio de UID + Ready state"""
        import time
        logger.info(f'⏳ Esperando a que el Pod se reinicie (cambio de UID)...')
        
        start_time = time.time()
        new_pod_detected = False
        
        # 1. Esperar a que cambie el UID (Pod recreado)
        while time.time() - start_time < timeout_seconds:
            current_uid = self._get_current_pod_uid()
            
            if current_uid and current_uid != old_uid:
                new_pod_detected = True
                logger.info(f"✅ Nuevo Pod detectado (UID: {current_uid})")
                break
            
            if not current_uid:
                logger.info("⏳ Pod antiguo terminado, esperando nuevo pod...")
            else:
                logger.info(f"⏳ Esperando terminación del pod antiguo ({current_uid})...")
            
            time.sleep(2)
            
        if not new_pod_detected:
            logger.error("❌ Timeout esperando recreación del pod")
            return False

        # 2. Esperar a que el nuevo pod esté Ready
        return self._wait_for_postgres_ready(timeout_seconds=timeout_seconds - (time.time() - start_time))

    def _wait_for_postgres_ready(self, timeout_seconds=600):
        """Espera a que PostgreSQL esté completamente ready después de un restart"""
        import subprocess
        import time
        
        logger.info(f'⏳ Esperando a que PostgreSQL esté ready (timeout={int(timeout_seconds)}s)...')
        
        # 1. Esperar a que el pod esté Ready
        try:
            cmd = [
                'kubectl', 'wait', 'pod',
                '-n', self.namespace,
                '-l', f'cnpg.io/cluster={self.cluster_name}',
                '--for=condition=Ready',
                f'--timeout={int(timeout_seconds)}s'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info('✅ Pod está Ready')
        except subprocess.CalledProcessError as e:
            logger.error(f'❌ Timeout esperando pod Ready: {e.stderr.decode()}')
            return False
        
        # 2. Esperar un poco más para asegurar que PostgreSQL acepta conexiones
        time.sleep(10)
        
        # 3. Verificar conexión con pg_isready
        max_attempts = 60
        for attempt in range(max_attempts):
            try:
                cmd = [
                    'kubectl', 'exec', '-n', self.namespace,
                    f'{self.cluster_name}-1',
                    '--', 'pg_isready', '-U', 'postgres'
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=5)
                if b'accepting connections' in result.stdout:
                    logger.info('✅ PostgreSQL está aceptando conexiones')
                    # Esperar estabilidad completa
                    logger.info('⏳ Esperando 20s adicionales para estabilidad...')
                    time.sleep(20)
                    return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt < max_attempts - 1:
                    logger.info(f"⏳ Esperando a PostgreSQL... ({attempt+1}/{max_attempts})")
                    time.sleep(5)
                    continue
                else:
                    logger.error('❌ PostgreSQL no responde después de restart')
                    return False
        
        return True

    # Hardware-determined constants: depend on storage type/deployment, not workload.
    # Injected into every patch so the agent doesn't waste exploration on known values.
    # Applied to ALL modes (training, Default baseline, PGTune baseline) — comparison is fair.
    #   random_page_cost=1.1, eio=32: SSD-optimal (USB SSD, conservative parallelism)
    #   wal_compression=on: reduces WAL volume ~30-40% on USB SSD, fewer write stalls.
    #   wal_buffers=16MB: PGTune recommendation; always ~16MB — postmaster param so
    #     making it tunable caused restart every step, corrupting benchmark measurements.
    #   max_wal_size=4GB: prevents checkpoint-during-benchmark at 150-400 TPS / 5-min run.
    #
    #   REMOVED (v11 C2 fix — TKDE submission):
    #   synchronous_commit=off: violates ACID durability (D in ACID). Not valid for academic
    #     comparison against OtterTune/LlamaTune/GPTuner which all use sc=on (PostgreSQL default).
    #   full_page_writes=off: affects crash-recovery integrity. Same category as sc=off —
    #     removing both keeps HARDWARE_CONSTANTS 100% defensible for reviewers.
    HARDWARE_CONSTANTS = {
        'random_page_cost': '1.1',
        'effective_io_concurrency': '32',
        'wal_compression': 'on',
        'wal_buffers': '16MB',
        'max_wal_size': '4GB',
        'min_wal_size': '1GB',
    }

    def __init__(self, namespace: str, cluster_name: str, tier: str = None):
        self.namespace = namespace
        self.cluster_name = cluster_name
        self._tier_constants = {}
        if tier:
            self.configure_tier(tier)

    def configure_tier(self, tier: str):
        """v11: shared_buffers is now in the RL action space — NOT injected as tier constant.
        The agent learns the optimal shared_buffers within the range defined in postgres_fog_v6.json.
        For Default/PGTune baselines, shared_buffers comes from get_default_config() / get_pgtune_config()
        and is no longer overridden here."""
        try:
            tier_key = tier if tier.startswith('rpi_') else f'rpi_{tier}'
            logger.info(f'[Applier] Tier={tier_key}: shared_buffers NOT fixed (v11: in action space)')
        except Exception as e:
            logger.warning(f'[Applier] configure_tier({tier}) failed: {e}')

    def apply_knobs(self, knobs: Dict[str, str]) -> Dict[str, any]:
        # Inject hardware constants (SSD-optimal + tier-specific shared_buffers)
        knobs.update(self.HARDWARE_CONSTANTS)
        knobs.update(self._tier_constants)  # shared_buffers at 25% RAM per tier
        # Convertir valores a formato con unidades
        knobs_with_units = self._add_units(knobs)
        
        # Obtener configuración actual para filtrar cambios reales
        try:
            current_knobs = self.get_current_knobs()
            # Filtrar solo lo que ha cambiado
            knobs_to_apply = {}
            for k, v in knobs_with_units.items():
                # Normalizar para comparación (str vs str)
                if str(current_knobs.get(k, '')) != str(v):
                    knobs_to_apply[k] = v
                else:
                    pass # logger.debug(f"Knob {k} no cambia ({v})")
            
            if not knobs_to_apply:
                logger.info("ℹ️  Ningún knob ha cambiado. Saltando patch.")
                return {'success': True, 'applied': {}, 'restart_required': False}
                
            knobs_with_units = knobs_to_apply
            logger.info(f"📝 Applying diff: {list(knobs_with_units.keys())}")
            
        except Exception as e:
            logger.warning(f"⚠️  Falló al obtener knobs actuales, aplicando todo: {e}")

        classified = PgKnobClassifier.classify_knobs(knobs_with_units)
        restart_required = len(classified['restart']) > 0
        
        print(f"📊 Clasificación (Diff):")
        print(f"   Restart: {list(classified['restart'].keys())}")
        print(f"   Reload: {list(classified['reload'].keys())}")
        
        # Obtener UID antes de aplicar cambio
        old_uid = self._get_current_pod_uid()
        if restart_required and not old_uid:
             logger.warning("⚠️ No se pudo obtener UID previo, la detección de restart puede fallar")

        try:
            self._patch_cluster(knobs_with_units)
            
            # Si requiere restart, esperar a que PostgreSQL esté ready
            if restart_required:
                logger.warning('⚠️  Knobs requieren restart, forzando recreación de Pod...')

                # Checkpoint antes de matar el pod — minimiza WAL replay en el restart
                self._force_checkpoint()

                # Force delete the pod to ensure restart happens immediately
                if old_uid:
                     self._force_delete_pod()
                
                if not self._wait_for_postgres_restart(old_uid):
                    return {'success': False, 'message': 'PostgreSQL no estuvo ready después del restart'}
            
            return {
                'success': True,
                'applied': knobs_with_units,
                'restart_required': restart_required
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def _force_checkpoint(self):
        """Fuerza un CHECKPOINT antes del restart para minimizar WAL recovery"""
        try:
            cmd = [
                'kubectl', 'exec', '-n', self.namespace,
                f'{self.cluster_name}-1',
                '--', 'psql', '-U', 'postgres', '-c', 'CHECKPOINT'
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            logger.info("✅ CHECKPOINT ejecutado — WAL recovery será mínimo tras restart")
        except Exception as e:
            logger.warning(f"⚠️ CHECKPOINT fallido (continuando): {e}")

    def _force_delete_pod(self):
        """Fuerza el borrado del pod para acelerar el restart"""
        try:
            logger.info(f"🗑️ Forzando borrado de pod en cluster {self.cluster_name}...")
            # Delete primary pod (CNPG will recreate it)
            # Find running primary pod
            cmd = [
                'kubectl', 'delete', 'pod',
                '-n', self.namespace,
                '-l', f'cnpg.io/cluster={self.cluster_name},cnpg.io/instanceRole=primary',
                '--grace-period=5'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("✅ Pod borrado. Esperando recreación...")
        except Exception as e:
            logger.warning(f"⚠️ Fallo al borrar pod: {e}")
    
    def _add_units(self, knobs: Dict[str, any]) -> Dict[str, str]:
        """Convierte valores numéricos a strings con unidades apropiadas"""
        result = {}
        
        for key, value in knobs.items():
            # Si ya tiene unidad (es string con letras), dejarlo
            if isinstance(value, str) and any(c.isalpha() for c in value):
                result[key] = value
                continue
            
            # Convertir a número
            try:
                num_value = float(value)
            except (ValueError, TypeError):
                result[key] = str(value)
                continue
            
            # Aplicar conversión según el tipo de knob
            if key in self.BLOCK_KNOBS:
                # Bloques de 8KB → MB
                mb_value = int(num_value * self.BLOCK_SIZE_KB / 1024)
                result[key] = f"{mb_value}MB"
            elif key in self.KB_KNOBS:
                # Keep KB precision — PostgreSQL accepts kB suffix.
                # Truncating to MB collapses work_mem to only 4 discrete values in 2GB tier.
                result[key] = f"{int(num_value)}kB"
            elif key in self.MB_KNOBS:
                result[key] = f"{int(num_value)}MB"
            elif key in self.SECONDS_KNOBS:
                result[key] = f"{int(round(num_value))}s"
            elif key in self.MS_KNOBS:
                result[key] = f"{int(round(num_value))}ms"
            elif key in self.INTEGER_KNOBS:
                result[key] = str(int(round(num_value)))
            elif key == 'synchronous_commit':
                # Continuous [0,1] → discrete on/off threshold at 0.5
                result[key] = 'off' if num_value < 0.5 else 'on'
            else:
                # Floats (checkpoint_completion_target, random_page_cost, etc.)
                result[key] = str(value)
        
        return result
    
    def _patch_cluster(self, knobs: Dict[str, str]):
        patch = {'spec': {'postgresql': {'parameters': knobs}}}
        cmd = [
            'kubectl', 'patch', 'cluster', self.cluster_name,
            '-n', self.namespace, '--type=merge',
            '-p', json.dumps(patch)
        ]
        print(f"🔧 Aplicando patch...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            raise Exception(f"Patch failed: {result.stderr}")
        
        print(f"✅ Patch aplicado")
    
    def get_current_knobs(self) -> Dict[str, str]:
        cmd = [
            'kubectl', 'get', 'cluster', self.cluster_name,
            '-n', self.namespace,
            '-o', 'jsonpath={.spec.postgresql.parameters}'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout) if result.stdout else {}

if __name__ == '__main__':
    applier = CnpgKnobApplier(namespace='pg', cluster_name='pg-cluster')
    
    # Test conversión de unidades
    test_knobs = {
        'shared_buffers': 32768,      # 32768 bloques * 8KB = 256MB
        'work_mem': 4096,              # 4096 KB = 4MB
        'effective_cache_size': 524288, # 524288 bloques * 8KB = 4096MB
        'checkpoint_completion_target': 0.5  # sin unidad
    }
    
    print(f"🧪 Test conversión:")
    converted = applier._add_units(test_knobs)
    for k, v in converted.items():
        print(f"   {k}: {test_knobs[k]} → {v}")
