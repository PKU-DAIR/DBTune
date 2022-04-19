import json
import time
import pandas as pd
import bisect
import numpy as np
from .utils import logger
import numpy as np
import os
import re
import pdb
import ast
ts = int(time.time())
logger = logger.get_logger('autotune', 'log/tune_database_{}.log'.format(ts))
INTERNAL_METRICS_LEN = 51
# Deprecated Var Definition
KNOBS = [
         # 'sync_binlog',
         # 'innodb_flush_log_at_trx_commit',
         # 'innodb_max_dirty_pages_pct',
         # 'innodb_io_capacity_max',
         # 'innodb_io_capacity',
         # 'innodb_max_dirty_pages_pct_lwm',
         # 'innodb_thread_concurrency',
         # 'innodb_lock_wait_timeout',
         # 'innodb_lru_scan_depth',
         #
         #'table_open_cache',
         #'innodb_buffer_pool_size',
         #'innodb_buffer_pool_instances',
         #'innodb_purge_threads',
         #'innodb_read_io_threads',
         #'innodb_write_io_threads',
         #'innodb_read_ahead_threshold',
         #'innodb_sync_array_size',
         #'innodb_sync_spin_loops',
         #'innodb_thread_concurrency',
         #'metadata_locks_hash_instances',
         #'innodb_adaptive_hash_index',
         #'tmp_table_size',
         #'innodb_random_read_ahead',
         #'table_open_cache_instances',
         #'thread_cache_size',
         #'innodb_io_capacity',
         #'innodb_lru_scan_depth',
         #'innodb_spin_wait_delay',
         #'innodb_adaptive_hash_index_parts',
         #'innodb_page_cleaners',
         #'innodb_flush_neighbors'
         #
         #'innodb_max_dirty_pages_pct',
         #'innodb_io_capacity_max',
         #'innodb_io_capacity',
         #'innodb_max_dirty_pages_pct_lwm',
         #'innodb_thread_concurrency',
         #'innodb_lock_wait_timeout',
         #'innodb_lru_scan_depth',
         #'innodb_buffer_pool_size',
         #'innodb_purge_threads',
         #'innodb_read_io_threads',
         #'innodb_write_io_threads',
         #'innodb_spin_wait_delay'
         #
         'innodb_max_dirty_pages_pct',
         #'innodb_io_capacity_max',
         'innodb_io_capacity',
         'innodb_max_dirty_pages_pct_lwm',
         'innodb_thread_concurrency',
         'innodb_lock_wait_timeout',
         'innodb_lru_scan_depth',
         #'innodb_buffer_pool_size',
         'innodb_buffer_pool_instances',
         'innodb_purge_threads',
         'innodb_read_io_threads',
         'innodb_write_io_threads',
         'innodb_spin_wait_delay',
         'table_open_cache',
         'binlog_cache_size',
         #'max_binlog_cache_size',
         'innodb_adaptive_max_sleep_delay',
         'innodb_change_buffer_max_size',
         'innodb_flush_log_at_timeout',
         'innodb_flushing_avg_loops',
         'innodb_max_purge_lag',
         'innodb_read_ahead_threshold',
         'innodb_sync_array_size',
         'innodb_sync_spin_loops',
         'metadata_locks_hash_instances',
         #'innodb_adaptive_hash_index',
         'tmp_table_size',
         #'innodb_random_read_ahead',
         'table_open_cache_instances',
         'thread_cache_size',
         'innodb_adaptive_hash_index_parts',
         'innodb_page_cleaners',
         'innodb_flush_neighbors',
]
KNOB_DETAILS = None
EXTENDED_KNOBS = None
num_knobs = len(KNOBS)


# Deprecated function
def init_knobs(num_total_knobs):
    global memory_size
    global disk_size
    global KNOB_DETAILS
    global EXTENDED_KNOBS

    # TODO: Test the request

    memory_size = 32 * 1024 * 1024 * 1024
    disk_size = 300 * 1024 * 1024 * 1024

    KNOB_DETAILS = {
        # 'sync_binlog': ['integer', [1, 1000, 1]],
        # 'innodb_flush_log_at_trx_commit': ['integer', [0, 2, 1]],
        # 'innodb_max_dirty_pages_pct': ['integer', [0, 99, 75]],
        # 'innodb_io_capacity_max': ['integer', [2000, 100000, 100000]],
        # 'innodb_io_capacity': ['integer', [100, 2000000, 20000]],
        # 'innodb_max_dirty_pages_pct_lwm': ['integer', [0, 99, 10]],
        # 'innodb_thread_concurrency': ['integer', [0, 10000, 0]],
        # 'innodb_lock_wait_timeout': ['integer', [1, 1073741824, 50]],
        # 'innodb_lru_scan_depth': ['integer', [100, 10240, 1024]],
        #
        #'table_open_cache': ['integer', [1, 10240, 512]],
        #'innodb_buffer_pool_size': ['integer', [1024 * 1024 * 1024, memory_size, 24 * 1024* 1024 * 1024]],
        #'innodb_buffer_pool_instances': ['integer', [1, 64, 8]],
        #'innodb_purge_threads': ['integer', [1, 32, 1]],
        #'innodb_read_io_threads': ['integer', [1, 64, 1]],
        #'innodb_write_io_threads': ['integer', [1, 64, 1]],
        #'innodb_read_ahead_threshold': ['integer', [0, 64, 56]],
        #'innodb_sync_array_size': ['integer', [1, 1024, 1]],
        #'innodb_sync_spin_loops': ['integer', [0, 100, 30]],
        #'innodb_thread_concurrency': ['integer', [0, 100, 16]],
        #'metadata_locks_hash_instances': ['integer', [1, 1024, 8]],
        #'innodb_adaptive_hash_index': ['boolean', ['ON', 'OFF']],
        #'tmp_table_size': ['integer', [1024, 1073741824, 1073741824]],
        #'innodb_random_read_ahead': ['boolean', ['ON', 'OFF']],
        #'table_open_cache_instances': ['integer', [1, 64, 16]],
        #'thread_cache_size': ['integer', [0, 1000, 512]],
        #'innodb_io_capacity': ['integer', [100, 2000000, 20000]],
        #'innodb_lru_scan_depth': ['integer', [100, 10240, 1024]],
        #'innodb_spin_wait_delay': ['integer', [0, 60, 6]],
        #'innodb_adaptive_hash_index_parts': ['integer', [1, 512, 8]],
        #'innodb_page_cleaners': ['integer', [1, 64, 4]],
        #'innodb_flush_neighbors': ['enum', [0, 2, 1]],
        #
        #'innodb_max_dirty_pages_pct': ['integer', [0, 99, 75]],
        #'innodb_io_capacity_max': ['integer', [2000, 100000, 100000]],
        #'innodb_io_capacity': ['integer', [100, 20000, 2000]],
        #'innodb_max_dirty_pages_pct_lwm': ['integer', [0, 99, 10]],
        #'innodb_thread_concurrency': ['integer', [0, 10000, 32]],
        #'innodb_lock_wait_timeout': ['integer', [1, 1073741824, 50]],
        #'innodb_lru_scan_depth': ['integer', [100, 10240, 1024]],
        #'innodb_buffer_pool_size': ['integer', [1024 * 1024 * 1024, memory_size, 8 * 1024* 1024 * 1024]],
        #'innodb_purge_threads': ['integer', [1, 32, 1]],
        #'innodb_read_io_threads': ['integer', [1, 64, 1]],
        #'innodb_write_io_threads': ['integer', [1, 64, 1]],
        #'innodb_spin_wait_delay': ['integer', [0, 60, 6]],
        #
        'innodb_max_dirty_pages_pct': ['integer', [0, 99, 75]],
        'innodb_io_capacity': ['integer', [100, 20000, 2000]],
        'innodb_max_dirty_pages_pct_lwm': ['integer', [0, 99, 10]],
        'innodb_thread_concurrency': ['integer', [0, 10000, 32]],
        'innodb_lock_wait_timeout': ['integer', [1, 1073741824, 50]],
        'innodb_lru_scan_depth': ['integer', [100, 10240, 1024]],
        #'innodb_buffer_pool_size': ['integer', [1024 * 1024 * 1024, memory_size, 8 * 1024* 1024 * 1024]],
        'innodb_buffer_pool_instances': ['integer', [1, 16, 8]],
        'innodb_purge_threads': ['integer', [1, 32, 1]],
        'innodb_read_io_threads': ['integer', [1, 64, 1]],
        'innodb_write_io_threads': ['integer', [1, 64, 1]],
        'innodb_spin_wait_delay': ['integer', [0, 60, 6]],
        'table_open_cache': ['integer', [1, 10240, 512]],
        'binlog_cache_size': ['integer', [4096, 4294967295, 32768]],
        'innodb_adaptive_max_sleep_delay': ['integer', [0, 10000000, 150000]],
        'innodb_change_buffer_max_size': ['integer', [0, 50, 25]], 
        'innodb_flush_log_at_timeout': ['integer', [1, 2700, 1]],
        'innodb_flushing_avg_loops': ['integer', [1, 1000, 30]],
        'innodb_max_purge_lag': ['integer', [0, 4294967295, 0]],
        'innodb_read_ahead_threshold': ['integer', [0, 64, 56]], 
        'innodb_sync_array_size': ['integer', [1, 1024, 1]],
        'innodb_sync_spin_loops': ['integer', [0, 100, 30]],
        'metadata_locks_hash_instances': ['integer', [1, 1024, 8]],
        #'innodb_adaptive_hash_index': ['boolean', ['ON', 'OFF']],
        'tmp_table_size': ['integer', [1024, 1073741824, 1073741824]],
        #'innodb_random_read_ahead': ['boolean', ['ON', 'OFF']],
        'table_open_cache_instances': ['integer', [1, 64, 16]],
        'thread_cache_size': ['integer', [0, 1000, 512]],
        'innodb_adaptive_hash_index_parts': ['integer', [1, 512, 8]],
        'innodb_page_cleaners': ['integer', [1, 64, 4]],
        'innodb_flush_neighbors': ['enum', [0, 2, 1]], 
    }
    # TODO: ADD Knobs HERE! Format is the same as the KNOB_DETAILS
    UNKNOWN = 0
    EXTENDED_KNOBS = {
        ##'thread_stack' : ['integer', [131072, memory_size, 524288]],
        #'back_log' : ['integer', [1, 65535, 900]],
    }
    # ADD Other Knobs, NOT Random Selected
    i = 0
    EXTENDED_KNOBS = dict(sorted(EXTENDED_KNOBS.items(), key=lambda d: d[0]))
    for k, v in EXTENDED_KNOBS.items():
        if i < num_total_knobs - num_knobs:
            KNOB_DETAILS[k] = v
            KNOBS.append(k)
            i += 1
        else:
            break


def gen_continuous(action):
    knobs = {}

    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]

        knob_type = value['type']

        if knob_type == 'integer':
            min_val, max_val = value['min'], value['max']
            delta = int((max_val - min_val) * action[idx])
            eval_value = min_val + delta 
            eval_value = max(eval_value, min_val)
            if value.get('stride'):
                all_vals = np.arange(min_val, max_val, value['stride'])
                indx = bisect.bisect_left(all_vals, eval_value)
                if indx == len(all_vals): indx -= 1
                eval_value = all_vals[indx]
            # TODO(Hong): add restriction among knobs, truncate, etc
            knobs[name] = eval_value
        elif knob_type == 'float':
            min_val, max_val = value['min'], value['max']
            delta = (max_val - min_val) * action[idx]
            eval_value = min_val + delta
            eval_value = max(eval_value, min_val)
            if value.get('stride'):
                all_vals = np.arange(min_val, max_val, value['stride'])
                indx = bisect.bisect_left(all_vals, eval_value)
                if indx == len(all_vals): indx -= 1
                eval_value = all_vals[indx]
            knobs[name] = eval_value
        elif knob_type == 'enum':
            enum_size = len(value['enum_values'])
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['enum_values'][enum_index]
            # TODO(Hong): add restriction among knobs, truncate, etc
            knobs[name] = eval_value
        elif knob_type == 'combination':
            enum_size = len(value['combination_values'])
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['combination_values'][enum_index]
            knobs_names = name.strip().split('|')
            knobs_value = eval_value.strip().split('|')
            for k, knob_name_tmp in enumerate(knobs_names):
                knobs[knob_name_tmp] = knobs_value[k]


    return knobs


def save_knobs(knobs, external_metrics):
    knob_json = json.dumps(knobs)
    result_str = '{},{},{},'.format(external_metrics[0], external_metrics[1], external_metrics[2])
    result_str += knob_json


def initialize_knobs(knobs_config, num):
    global KNOBS
    global KNOB_DETAILS
    if num == -1:
        f = open(knobs_config)
        KNOB_DETAILS = json.load(f)
        KNOBS = list(KNOB_DETAILS.keys())
        f.close()
    else:
        f = open(knobs_config)
        knob_tmp = json.load(f)
        i = 0
        KNOB_DETAILS = {}
        while i < num:
            key = list(knob_tmp.keys())[i]
            KNOB_DETAILS[key] = knob_tmp[key]
            i = i + 1
        KNOBS = list(KNOB_DETAILS.keys())
        f.close()
    return KNOB_DETAILS


def get_default_knobs():
    default_knobs = {}
    for name, value in KNOB_DETAILS.items():
        if not value['type'] == "combination":
            default_knobs[name] = value['default']
        else:
            knobL = name.strip().split('|')
            valueL = value['default'].strip().split('|')
            for i in range(0, len(knobL)):
                default_knobs[knobL[i]] = int(valueL[i])
    return default_knobs


def get_knob_details(knobs_config):
    initialize_knobs(knobs_config)
    return KNOB_DETAILS


def knob2action(knob):
    actionL = []
    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]
        knob_type = value['type']
        if knob_type == "enum":
            enum_size = len(value['enum_values'])
            action = value['enum_values'].index(knob[name]) / enum_size
        else:
            min_val, max_val = value['min'], value['max']
            action = (knob[name] - min_val) / (max_val - min_val)

        actionL.append(action)

    return np.array(actionL)

def knobDF2action(df):
    actionL =  pd.DataFrame(np.zeros_like(df),columns=df.columns)
    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]
        if value['type'] in  ["integer", "float"]:
            min_val, max_val = value['min'], value['max']
            action = (df[name] - min_val) / (max_val - min_val)
            actionL[name] = action
        if value['type'] ==  "enum":
            actionL[name]=''
            enum_size = len(value['enum_values'])
            for i in range(0, df.shape[0]):
                actionL.loc[i, name] = value['enum_values'].index(str(df[name].iloc[i])) / enum_size
        if value['type'] == "combination":
            actionL[name] = ''
            combination_size = len(value['combination_values'])
            combination_knobs = name.strip().split('|')
            for i in range(0, df.shape[0]):
                combination_value = ""
                for knob in combination_knobs:
                    if combination_value == "":
                        combination_value = str(df[knob].iloc[i])
                    else:
                        combination_value = combination_value + "|" + str(df[knob].iloc[i])
                actionL.loc[i, name] = value['combination_values'].index(combination_value) / combination_size

    return np.array(actionL)




def get_data_for_mapping(fn):
    '''
    get konbs and tps from res file.
    Only those that meet the JSON requirements are returned. If none suitable, return FALSE
    '''
    f = open(fn)
    lines = f.readlines()
    f.close()
    konbL = []
    tpsL = []
    internal_metricL = []
    if os.path.getsize(fn) == 0:
        return False

    #check whether KNOBS is contained in the res file, for further mapping
    line = lines[0]
    t = line.strip().split('|')
    knob_str = t[0]
    knob_str = re.sub(r"[A-Z]+", "1", knob_str)
    tmp = re.findall(r"\D*", knob_str)
    old_knob = [name.strip('_') for name in tmp if name not in ['', '.']]
    combinationL = []

    for knob_name in KNOBS:
        if '|' in knob_name: #deal with combination type
            knobs = knob_name.split('|')
            combinationL.append(knob_name)
            for knob in knobs:
                if not knob in old_knob:
                    return False
        else:
            if not knob_name in old_knob:
                return False

    for line in lines:
        t = line.strip().split('|')
        knob_str = t[0]
        valueL_tmp = re.findall('(\d+(?:\.\d+)?)', knob_str)
        valueL = []
        for item in valueL_tmp:
            if item.isdigit():
                valueL.append(int(item))
            else:
                try:
                    valueL.append(float(item))
                except:
                    valueL.append(item)
        knob_str = re.sub(r"[A-Z]+", "1", knob_str)
        tmp = re.findall(r"\D*", knob_str)
        nameL = [name.strip('_') for name in tmp if name not in ['', '.']]
        tps = float(t[1])

        internal_metric = ast.literal_eval(t[4])
        if not len(internal_metric) == INTERNAL_METRICS_LEN:
            if len(t) == 5:
                del(internal_metric[51:65])
            elif t[5] == '65d':
                from .utils.parser import convert_65IM_to_51IM
                internal_metric = list(convert_65IM_to_51IM(np.array(internal_metric)))

        if len(combinationL):
            for name in combinationL:
                value = KNOB_DETAILS[name]
                combination_knobs = name.strip().split('|')
                combination_value = ""
                for knob in combination_knobs:
                    if combination_value == "":
                        combination_value = str(valueL[nameL.index(knob)])
                    else:
                        combination_value = combination_value + "|" + str(valueL[nameL.index(knob)])

                if not combination_value in value['combination_values']:
                    # the combination value is not in the range appointed in json, abort that row
                    continue

        konbL.append(valueL)
        tpsL.append(tps)
        internal_metricL.append(internal_metric)


    if len(tpsL) ==0:
        return False

    knob_df = pd.DataFrame(konbL, columns=nameL)
    internal_metricM = np.vstack(internal_metricL)

    #logger.info("51 metrics: {}".format(fn))
    return knob_df, tpsL, internal_metricM




def gen_continuous_one_hot(action):
    knobs = {}
    action_idx = 0

    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]

        knob_type = value['type']

        if knob_type == 'integer':
            min_val, max_val = value['min'], value['max']
            delta = int((max_val - min_val) * action[action_idx])
            eval_value = min_val + delta
            eval_value = max(eval_value, min_val)
            if value.get('stride'):
                all_vals = np.arange(min_val, max_val, value['stride'])
                indx = bisect.bisect_left(all_vals, eval_value)
                if indx == len(all_vals): indx -= 1
                eval_value = all_vals[indx]
            # TODO(Hong): add restriction among knobs, truncate, etc
            knobs[name] = eval_value
            action_idx = action_idx + 1
        if knob_type == 'float':
            min_val, max_val = value['min'], value['max']
            delta = (max_val - min_val) * action[action_idx]
            eval_value = min_val + delta
            eval_value = max(eval_value, min_val)
            all_vals = np.arange(min_val, max_val, value['stride'])
            indx = bisect.bisect_left(all_vals, eval_value)
            if indx == len(all_vals): indx -= 1
            eval_value = all_vals[indx]
            knobs[name] = eval_value
            action_idx = action_idx + 1
        elif knob_type == 'enum':
            enum_size = len(value['enum_values'])
            feature = action[action_idx : action_idx + enum_size]
            enum_index = feature.argmax()
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['enum_values'][enum_index]
            # TODO(Hong): add restriction among knobs, truncate, etc
            knobs[name] = eval_value
            action_idx = action_idx + enum_size
        elif knob_type == 'combination':
            enum_size = len(value['combination_values'])
            enum_index = int(enum_size * action[action_idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['combination_values'][enum_index]
            knobs_names = name.strip().split('|')
            knobs_value = eval_value.strip().split('|')
            for k, knob_name_tmp in enumerate(knobs_names):
                knobs[knob_name_tmp] = knobs_value[k]
            action_idx = action_idx + 1


    return knobs


def knobDF2action_onehot(df):
    feature_len = 0
    for k in KNOB_DETAILS.keys():
        if KNOB_DETAILS[k]['type'] == 'enum':
            feature_len = feature_len + len(KNOB_DETAILS[k]['enum_values'])
        else:
            feature_len = feature_len + 1

    actionL_all = []
    for i in range(df.shape[0]):
        actionL = []
        for j in range(df.shape[1]):
            knob = df.columns[j]
            value = KNOB_DETAILS[knob]
            if value['type'] in ["integer", "float"]:
                min_val, max_val = value['min'], value['max']
                action = (df[knob].iloc[i] - min_val) / (max_val - min_val)
                actionL.append(action)
            else:
                for tmp in value['enum_values']:
                    if tmp == df[knob].iloc[i]:
                        actionL.append(1)
                    else:
                        actionL.append(0)
        actionL_all.append(actionL)

    return np.hstack(actionL_all).reshape(-1, feature_len)
