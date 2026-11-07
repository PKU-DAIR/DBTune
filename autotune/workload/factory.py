"""
Factory para crear WorkloadRunners según configuración
"""
from typing import Dict
from autotune.workload.k8s_tpcc_runner import K8sTpccRunner
from autotune.workload.k8s_tpch_runner import K8sTpchRunner


class WorkloadFactory:
    """Factory para crear el runner apropiado según benchmark type"""
    
    @staticmethod
    def create_runner(benchmark_type: str, namespace: str = 'benchmark'):
        """
        Crea runner según tipo de benchmark
        
        Args:
            benchmark_type: 'tpcc', 'tpch'
            namespace: Kubernetes namespace
            
        Returns:
            WorkloadRunner apropiado
        """
        benchmark_type = benchmark_type.lower()
        
        if benchmark_type == 'tpcc':
            return K8sTpccRunner(
                namespace=namespace,
                config_name='hammerdb-scripts'
            )
        elif benchmark_type == 'tpch':
            return K8sTpchRunner(
                namespace=namespace,
                config_name='hammerdb-pg-tpch-scripts'
            )
        else:
            raise ValueError(f"Benchmark type '{benchmark_type}' no soportado. Use 'tpcc' o 'tpch'")
    
    @staticmethod
    def get_default_params(benchmark_type: str) -> Dict:
        """Retorna parámetros por defecto según benchmark"""
        if benchmark_type.lower() == 'tpcc':
            return {
                'vus': 20,
                'rampup_min': 2,
                'duration_min': 5
            }
        elif benchmark_type.lower() == 'tpch':
            return {
                'vus': 1,
                'querysets': 1,
                'dop': 2,
                'scale': 1
            }
        else:
            raise ValueError(f"Benchmark type '{benchmark_type}' no soportado")


# Función helper para simplificar uso
def run_benchmark(benchmark_type: str, namespace: str = 'benchmark', **kwargs) -> Dict[str, float]:
    """
    Helper function para ejecutar benchmark con un solo llamado
    
    Args:
        benchmark_type: 'tpcc' o 'tpch'
        namespace: Kubernetes namespace
        **kwargs: Parámetros específicos del benchmark
        
    Returns:
        Dict con métricas del benchmark
    """
    runner = WorkloadFactory.create_runner(benchmark_type, namespace)
    
    # Si no hay kwargs, usar defaults
    if not kwargs:
        kwargs = WorkloadFactory.get_default_params(benchmark_type)
    
    return runner.run(**kwargs)
