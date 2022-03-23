#! /bin/bash
# Usage: ./gen_mean_var_file.sh ./log/train_ddpg_1573239791.log
grep "internal metrics" $1 > /tmp/normalize_log
python normalize.py /tmp/normalize_log
