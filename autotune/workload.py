from dynaconf import settings
# NUM_TABLES = 16
# TABLE_RANGE = 200000
SYSBENCH_WORKLOAD = {
    'name': 'sysbench',
    'type': 'read',
    # bash run_sysbench.sh write localhost 3318 root ''  theads table   32 150 output.log sbtest
    'cmd': 'bash {} {} {} {} {} "" {} {} {} {} {} {} {}'
}


OLTPBENCH_WORKLOADS = {
    'name': 'oltpbench',
    'type': 'oltpbenchmark',
    # bash run_oltpbench.sh benchmark config_xml output_file
    'cmd': 'bash {} {} {} {}'
}


JOB_WORKLOAD = {
    'name': 'job',
    'type': 'read',
    # bash run_job.sh queries_list.txt query_dir output.log MYSQL_SOCK
    'cmd': 'bash {} {} {} {} {}'
}


TPCH_WORKLOAD = {
    'name': 'tpch',
    'type': 'read',
    # bash run_job.sh queries_list.txt query_dir output.log MYSQL_SOCK
    'cmd': 'bash {} {} {} {} {}'
}
