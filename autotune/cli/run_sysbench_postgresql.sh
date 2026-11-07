#!/usr/bin/env bash

if [ -z "$SYSBENCH_BIN" ]; then
    SYSBENCH_BIN=$(which sysbench)
fi

if [ "${1}" == "read" ]
then
    run_script="oltp_read_only"
elif [ "${1}" == "write" ]
then
    run_script="oltp_write_only"
else
    run_script="oltp_read_write"
fi

${SYSBENCH_BIN} ${run_script} \
        --db-driver=pgsql \
        --pgsql-host=$2 \
        --pgsql-port=$3 \
        --pgsql-user=$4 \
        --pgsql-password=$5 \
        --pgsql-db=${12} \
        --range-size=100 \
        --events=0 \
        --rand-type=uniform \
        --tables=$6 \
        --table-size=$7 \
        --report-interval=1 \
        --warmup-time=$8 \
        --threads=$9 \
        --time=${10} \
        run > ${11}
