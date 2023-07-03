#!/usr/bin/env bash

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
        --db-ps-mode=disable \
        --report-interval=1 \
        --warmup-time=$8 \
        --threads=$9 \
        --time=${10} \
        --db-ps-mode=disable \
        run > ${11}
