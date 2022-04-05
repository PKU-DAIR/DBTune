#!/usr/bin/env bash
oltpbenchmark -b ${1} -c ${2} --execute=true -s 1 -o ${3}
