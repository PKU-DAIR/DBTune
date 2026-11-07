"""
Kubernetes TPC-H WorkloadRunner usando HammerDB Jobs
"""
import re
import time
import subprocess
from typing import Dict


class K8sTpchRunner:
    def __init__(self, namespace='benchmark', config_name='hammerdb-pg-tpch-scripts'):
        self.namespace = namespace
        self.config_name = config_name
        
    def run(self, vus: int = 1, querysets: int = 1, dop: int = 2, scale: int = 1) -> Dict[str, float]:
        """
        Ejecuta benchmark TPC-H y retorna métricas
        
        Args:
            vus: Número de virtual users (normalmente 1 para TPC-H)
            querysets: Número de query sets a ejecutar
            dop: Degree of parallelism
            scale: Scale factor (1, 10, 100, etc.)
            
        Returns:
            Dict con 'qph' (queries per hour) y 'geomean_sec'
        """
        job_name = f"tpch-run-{int(time.time())}"
        
        try:
            # 1. Crear Job
            self._create_job(job_name, vus, querysets, dop, scale)
            
            # 2. Esperar completación (TPC-H tarda más)
            timeout_sec = querysets * 180  # ~3 min por query set con scale 1
            self._wait_for_completion(job_name, timeout_sec)
            
            # 3. Parsear resultados
            results = self._parse_results(job_name)
            
            return results
            
        finally:
            # 4. Cleanup
            self._cleanup_job(job_name)
    
    def _create_job(self, job_name: str, vus: int, querysets: int, dop: int, scale: int):
        """Crea Job de Kubernetes"""
        job_yaml = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {self.namespace}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        role: benchmark
      containers:
      - name: hammerdb
        image: tpcorg/hammerdb:latest
        env:
        - name: PG_HOST
          value: "pg-cluster-rw.pg.svc.cluster.local"
        - name: PG_PORT
          value: "5432"
        - name: PG_SSLMODE
          value: "prefer"
        - name: PG_DB
          value: "tpch"
        - name: TPCH_SCALE
          value: "{scale}"
        - name: TPCH_TOTAL_QUERYSETS
          value: "{querysets}"
        - name: TPCH_DOP
          value: "{dop}"
        - name: TPCH_VU
          value: "{vus}"
        envFrom:
        - secretRef:
            name: pg-bench-secret
        volumeMounts:
        - name: scripts
          mountPath: /work
        command: ["./hammerdbcli", "auto", "/work/pg_tpch_run.tcl"]
      volumes:
      - name: scripts
        configMap:
          name: {self.config_name}
"""
        
        cmd = ['kubectl', 'apply', '-f', '-']
        subprocess.run(cmd, input=job_yaml.encode(), check=True, capture_output=True)
        print(f"✅ Job {job_name} creado")
    
    def _wait_for_completion(self, job_name: str, timeout_sec: int):
        """Espera a que Job complete"""
        print(f"⏳ Esperando completación de {job_name} (timeout={timeout_sec}s)...")
        
        cmd = [
            'kubectl', 'wait', '--for=condition=complete',
            f'job/{job_name}', '-n', self.namespace,
            f'--timeout={timeout_sec}s'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Job {job_name} no completó: {result.stderr}")
        
        print(f"✅ Job {job_name} completado")
    
    def _parse_results(self, job_name: str) -> Dict[str, float]:
        """Parsea logs y extrae QPH y geometric mean"""
        # Obtener logs
        cmd = ['kubectl', 'logs', '-n', self.namespace, f'job/{job_name}']
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        # Decodificar con manejo de errores
        logs = result.stdout.decode('utf-8', errors='ignore')
        
        # Buscar: "Completed 1 query set(s) in 31 seconds"
        pattern_time = r'Completed (\d+) query set\(s\) in (\d+) seconds'
        match_time = re.search(pattern_time, logs)
        
        # Buscar: "Geometric mean of query times returning rows (22) is 0.94304"
        pattern_geomean = r'Geometric mean of query times returning rows \(\d+\) is ([\d.]+)'
        match_geomean = re.search(pattern_geomean, logs)
        
        if not match_time:
            raise ValueError(f"No se encontró tiempo de completado en logs del job {job_name}")
        
        querysets = int(match_time.group(1))
        total_seconds = int(match_time.group(2))
        
        # Calcular QPH (Queries Per Hour)
        # TPC-H tiene 22 queries por set
        total_queries = querysets * 22
        qph = (total_queries * 3600) / total_seconds
        
        # Geometric mean (opcional)
        geomean_sec = float(match_geomean.group(1)) if match_geomean else 0.0
        
        print(f"📊 Resultados: QPH={qph:.2f}, Geometric Mean={geomean_sec:.3f}s")
        
        return {
            'qph': qph,
            'geomean_sec': geomean_sec,
            'total_seconds': float(total_seconds)
        }
    
    def _cleanup_job(self, job_name: str):
        """Elimina Job"""
        cmd = ['kubectl', 'delete', 'job', job_name, '-n', self.namespace, '--ignore-not-found=true']
        subprocess.run(cmd, capture_output=True)
        print(f"🗑️  Job {job_name} eliminado")


if __name__ == '__main__':
    # Test
    runner = K8sTpchRunner()
    results = runner.run(vus=1, querysets=1, dop=2, scale=1)
    print(f"\n✅ Test completado: {results}")
