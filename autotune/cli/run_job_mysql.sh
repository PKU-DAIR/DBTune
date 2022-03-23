#!/usr/bin/env bash
# run_job_mysql.sh  selectedList.txt  queries_dir   output	MYSQL_SOCK

printf "query\tlat(ms)\n" > $3

while read a
  do
  {
  tmp=$(mysql -uroot -S$4 imdbload < $2/$a | tail -n 1 )
  query=`echo $tmp | awk '{print $1}'`
  lat=`echo $tmp | awk '{print $2}'`

  printf "$query\t$lat\n" >> $3
  } &
done < $1
