#!/usr/bin/env bash
# run_job_mysql.sh  selectedList.txt  queries_dir   output	MYSQL_SOCK

printf "query\tlat(ms)\n" > $3

while read a
  do
  {
    query=`echo ${a%.*}`
    start=$(date +%s%N)
    result=`psql imdbload < $2/$a`
    if [[ -z $result ]];
    then
      lat=240000
    else
      end=$(date +%s%N)
      lat=$(( ($end - $start) / 1000000 ))
    fi
  printf "$query\t$lat\n" >> $3
  } &
done < $1
