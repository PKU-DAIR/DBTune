# Workload Preparation 

DBTune currently supports three database benchmarks:  <a href="https://github.com/oltpbenchmark/oltpbench.git" target="_blank" rel="nofollow">OLTP\-Bench</a>,  <a href="https://github.com/akopytov/sysbench.git" target="_blank" rel="nofollow">SYSBENCH</a>  and <a href="https://github.com/winkyao/join-order-benchmark" target="_blank" rel="nofollow">JOB</a>. 

## SYSBENCH

Download and install

```shell
git clone https://github.com/akopytov/sysbench.git
./autogen.sh
./configure
make && make install
```

Load data

```shell
sysbench --db-driver=mysql --mysql-host=$HOST --mysql-socket=$SOCK --mysql-port=$MYSQL_PORT --mysql-user=root --mysql-password=$PASSWD --mysql-db=sbtest --table_size=800000 --tables=150 --events=0 --threads=32 oltp_read_write prepare > sysbench_prepare.out
```



## OLTP-Bench

We install OLTP-Bench to use the following workload: TPC-C, SEATS, Smallbank, TATP, Voter, Twitter, SIBench.

- Download

```
git clone https://github.com/oltpbenchmark/oltpbench.git
```

- To run `oltpbenchmark` outside the folder, modify the following file:

  - ./src/com/oltpbenchmark/DBWorkload.java (Line 85)

    ```shell

    pluginConfig = new XMLConfiguration("PATH_TO_OLTPBENCH/config/plugin.xml"); # modify this

    ```

  - ./oltpbenchmark

    ```

    #!/bin/bash

    java -Xmx8G -cp `$OLTPBENCH_HOME/classpath.sh bin` -Dlog4j.configuration=$OLTPBENCH_HOME/log4j.properties com.oltpbenchmark.DBWorkload $@

    ```

  - ./classpath.sh

    ```shell

    #!/bin/bash

    echo -ne "$OLTPBENCH_HOME/build"

    for i in `ls $OLTPBENCH_HOME/lib/*.jar`; do

        # IMPORTANT: Make sure that we do not include hsqldb v1

        if [[ $i =~ .*hsqldb-1.* ]]; then

            continue

        fi

        echo -ne ":$i"

    done

    ```

- Install 

  ```shell
  ant bootstrap
  ant resolve
  ant build
  ```



## Join-Order-Benchmark (JOB)

Download IMDB Data Set from http://homepages.cwi.nl/~boncz/job/imdb.tgz.

Follow the instructions of https://github.com/winkyao/join-order-benchmark to load data into MySQL.




