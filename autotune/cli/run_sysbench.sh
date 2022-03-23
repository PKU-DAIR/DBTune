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
        --mysql-host=$2 \
        --mysql-port=$3 \
        --mysql-user=$4 \
        --mysql-password=$5 \
        --mysql-socket=$MYSQL_SOCK \
        --mysql-db=${12} \
        --db-driver=mysql \
        --mysql-storage-engine=innodb \
        --range-size=100 \
        --events=0 \
        --rand-type=uniform \
        --tables=$6 \
        --table-size=$7 \
        --db-ps-mode=disable \
        --report-interval=10 \
        --warmup-time=$8 \
        --threads=$9 \
        --time=${10} \
        --db-ps-mode=disable \
        run > ${11}
