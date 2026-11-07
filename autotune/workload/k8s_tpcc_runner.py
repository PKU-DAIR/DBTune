"""
Kubernetes TPC-C WorkloadRunner usando HammerDB Jobs
"""
import re
import time
import subprocess
from typing import Dict, Optional


class K8sTpccRunner:
    def __init__(self, namespace='benchmark', config_name='hammerdb-scripts'):
        self.namespace = namespace
        self.config_name = config_name
        
    def run(self, vus: int, rampup_min: int, duration_min: int) -> Dict[str, float]:
        """
        Ejecuta benchmark TPC-C y retorna métricas
        
        Args:
            vus: Número de virtual users
            rampup_min: Tiempo de calentamiento en minutos
            duration_min: Duración del test en minutos
            
        Returns:
            Dict con 'nopm' y 'tpm'
        """
        job_name = f"tprocc-run-{int(time.time())}"
        
        try:
            # 1. Crear Job
            self._create_job(job_name, vus, rampup_min, duration_min)
            
            # 2. Esperar completación
            self._wait_for_completion(job_name, timeout_sec=(rampup_min + duration_min + 10) * 60)
            
            # 3. Parsear resultados
            results = self._parse_results(job_name)
            
            return results
            
        finally:
            # 4. Cleanup
            self._cleanup_job(job_name)
    
    def _create_job(self, job_name: str, vus: int, rampup_min: int, duration_min: int):
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
        - name: PGHOST
          value: "pg-cluster-rw.pg.svc.cluster.local"
        - name: PGPORT
          value: "5432"
        - name: TPCC_VUS
          value: "{vus}"
        - name: TPCC_RAMPUP_MIN
          value: "{rampup_min}"
        - name: TPCC_DURATION_MIN
          value: "{duration_min}"
        envFrom:
        - secretRef:
            name: pg-bench-secret
        volumeMounts:
        - name: scripts
          mountPath: /work
        command: ["./hammerdbcli", "auto", "/work/pg_tprocc_run.tcl"]
      volumes:
      - name: scripts
        configMap:
          name: {self.config_name}
"""
        
        # Aplicar con kubectl
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
        """Parsea logs y extrae NOPM y TPM"""
        # Obtener logs (sin text=True para manejar encoding)
        cmd = ['kubectl', 'logs', '-n', self.namespace, f'job/{job_name}']
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        # Decodificar con manejo de errores para símbolo copyright
        logs = result.stdout.decode('utf-8', errors='ignore')
        
        # Buscar: "TEST RESULT : System achieved 25801 NOPM from 59948 PostgreSQL TPM"
        pattern = r'TEST RESULT : System achieved (\d+) NOPM from (\d+) PostgreSQL TPM'
        match = re.search(pattern, logs)
        
        if not match:
            print(f"❌ Output completo del job {job_name}:\n{logs[-3000:]}")
            raise ValueError(f"No se encontró TEST RESULT en logs del job {job_name}")
        
        nopm = float(match.group(1))
        tpm = float(match.group(2))
        
        print(f"📊 Resultados: NOPM={nopm}, TPM={tpm}")
        
        return {
            'nopm': nopm,
            'tpm': tpm
        }
    
    def _cleanup_job(self, job_name: str):
        """Elimina Job"""
        cmd = ['kubectl', 'delete', 'job', job_name, '-n', self.namespace, '--ignore-not-found=true']
        subprocess.run(cmd, capture_output=True)
        print(f"🗑️  Job {job_name} eliminado")


if __name__ == '__main__':
    # Test
    runner = K8sTpccRunner()
    results = runner.run(vus=10, rampup_min=1, duration_min=2)
    print(f"\n✅ Test completado: {results}")
