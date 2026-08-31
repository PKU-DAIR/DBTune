#!/usr/bin/env bash
cd /work/dpl-sfc/users/tatsu/tmp/DBTune/third_party/oltpbench
./oltpbenchmark -b ${1} -c ${2} --execute=true -s 1 -o ${3}
mkdir -p /work/dpl-sfc/users/tatsu/tmp/DBTune/scripts/results/
mv results/* /work/dpl-sfc/users/tatsu/tmp/DBTune/scripts/results/