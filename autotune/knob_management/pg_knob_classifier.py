"""
Clasificador de PostgreSQL knobs según tipo de restart requerido
"""

class PgKnobClassifier:
    """Clasifica knobs según tipo de restart requerido"""
    
    POSTMASTER_KNOBS = {
        'max_connections', 'shared_buffers', 'max_worker_processes',
        'wal_buffers', 'wal_level', 'huge_pages'
    }
    
    SIGHUP_KNOBS = {
        'work_mem', 'maintenance_work_mem', 'effective_cache_size',
        'random_page_cost', 'seq_page_cost', 'effective_io_concurrency',
        'max_wal_size', 'min_wal_size', 'checkpoint_completion_target',
        'checkpoint_timeout', 'wal_compression',
        'synchronous_commit', 'bgwriter_lru_maxpages', 'wal_writer_delay',
        'bgwriter_delay', 'bgwriter_flush_after',
    }
    
    @classmethod
    def requires_restart(cls, knob_name: str) -> bool:
        return knob_name in cls.POSTMASTER_KNOBS
    
    @classmethod
    def requires_reload(cls, knob_name: str) -> bool:
        return knob_name in cls.SIGHUP_KNOBS
    
    @classmethod
    def classify_knobs(cls, knobs: dict) -> dict:
        result = {'restart': {}, 'reload': {}, 'unknown': {}}
        for knob, value in knobs.items():
            if cls.requires_restart(knob):
                result['restart'][knob] = value
            elif cls.requires_reload(knob):
                result['reload'][knob] = value
            else:
                result['unknown'][knob] = value
        return result
