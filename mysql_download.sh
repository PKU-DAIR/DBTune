#! /bin/bash

cd $(dirname $0)

mkdir -p mysql_build/data
cd mysql_build
mkdir -p cnf
ROOT="$(pwd)"
cat > cnf/my.cnf << EOF
[mysqld]
basedir = ${ROOT}
datadir = ${ROOT}/data
socket = ${ROOT}/mysql.sock
pid-file = ${ROOT}/mysql.pid
log-error = ${ROOT}/mysql.err
EOF

if [ ! -d boost ]; then
	mkdir boost
	wget https://archives.boost.io/release/1.77.0/source/boost_1_77_0.tar.bz2
	tar --bzip2 -xf boost_1_77_0.tar.bz2 -C boost --strip-components=1
fi

cmake ../third_party/mysql-server \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DWITH_BUILD_ID=0 \
    -DWITH_ASAN=0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_BOOST=./boost \
    -DWITHOUT_EXAMPLE_STORAGE_ENGINE=1 \
    -DWITHOUT_FEDERATED_STORAGE_ENGINE=1 \
    -DWITHOUT_ARCHIVE_STORAGE_ENGINE=1 \
    -DWITHOUT_BLACKHOLE_STORAGE_ENGINE=1 \
    -DWITHOUT_NDB_STORAGE_ENGINE=1 \
    -DWITHOUT_NDBCLUSTER_STORAGE_ENGINE=1 \
    -DWITHOUT_PARTITION_STORAGE_ENGINE=1 \
    -G Ninja
ninja $1 -j `nproc`
