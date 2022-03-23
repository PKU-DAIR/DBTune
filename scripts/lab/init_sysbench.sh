#! /bin/bash
# create tpcc database
MYSQL_BIN=mysql
RDS_HOST='localhost'
MYSQL_PORT=3306
SOCK='/data1/ruike/mysql/base/mysql.sock'
DBNAME=sbtest
TABLE_SIZE=800000
TABLES=150
SYSBENCH_HOME=/data1/ruike/sysbench

$MYSQL_BIN -uroot -S$SOCK -h $RDS_HOST -P$MYSQL_PORT -e "drop database if exists $DBNAME"
$MYSQL_BIN -uroot -S$SOCK -h $RDS_HOST -P$MYSQL_PORT -e "create database $DBNAME"
$SYSBENCH_HOME/src/sysbench --db-driver=mysql --mysql-host=$RDS_HOST --mysql-port=$MYSQL_PORT --mysql-user=root --mysql-socket=$SOCK --mysql-db=$DBNAME --table_size=$TABLE_SIZE --tables=$TABLES --events=0 --threads=32 oltp_read_write prepare > /tmp/sysbench_prepare.out
