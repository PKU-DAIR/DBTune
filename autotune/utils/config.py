# -*- coding: utf-8 -*-

import configparser
import json
from collections import defaultdict
from autotune.utils.history_container import detect_valid_history_file

class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d



knob_config = {}


default_value = {
'isolation_mode': 'False',
'online_mode': 'False',
'remote_mode': 'True',
'pid':0,
'max_runs': 200,
'knob_num': 'auto',
'selector_type': 'shap',
'initial_runs' : 10,
'initial_tunable_knob_num': 'auto',
'incremental': 'none',
'incremental_every' : 'auto',
'incremental_num':5,
'optimize_method': 'SMAC',
'tr_init': 'True',
'batch_size':16,
'transfer_framework':'auto',
'data_repo': 'DBTune_history'
}

auto_setting = ['knob_num', 'initial_tunable_knob_num', 'incremental_every',  'transfer_framework']

def get_default_dict(dic):
    config_dic =  defaultdict(str)
    for k in dic:
        config_dic[k] = dic[k]
    for key in default_value.keys():
        if key not in config_dic.keys() or config_dic[key] == '':
            config_dic[key] = default_value[key]
    for key in auto_setting:
        if config_dic[key] == 'auto':
            if key == 'knob_num':
                if  len(knob_config.keys()) < 40:
                    config_dic['knob_num'] = len(knob_config.keys())
                else:
                    config_dic['knob_num'] = 40
            if key == 'initial_tunable_knob_num':
                if config_dic['incremental'].lower() == 'decrease':
                    config_dic['initial_tunable_knob_num'] =  config_dic['knob_num']
                elif config_dic['incremental'].lower() == 'increase':
                    config_dic['initial_tunable_knob_num'] = 5
                else:
                    config_dic['initial_tunable_knob_num'] = int(int(config_dic['knob_num'])/2)
            if key ==  'incremental_every':
                if config_dic['incremental'].lower() == 'decrease':
                    config_dic['incremental_every'] =  int(config_dic['max_runs'] / (config_dic['initial_tunable_knob_num'] / config_dic['incremental_num'])) + 1
                elif config_dic['incremental'].lower() == 'increase':
                    config_dic['incremental_every'] = int(config_dic['max_runs'] / (( config_dic['knob_num'] - config_dic['initial_tunable_knob_num']) / config_dic['incremental_num']))

                else:
                    config_dic['incremental_every'] = 0
            if key == 'transfer_framework':
                if len(detect_valid_history_file(config_dic['data_repo'])) > 0:
                    config_dic['transfer_framework'] = 'rgpe'
                else:
                    config_dic['transfer_framework'] = 'none'





    return config_dic

def parse_args(file):
    cf = DictParser()
    cf.read(file, encoding="utf-8")
    config_dict = cf.read_dict()
    global knob_config
    f = open(config_dict['database']['knob_config_file'])
    knob_config = json.load(f)

    return get_default_dict(config_dict["database"]), get_default_dict(config_dict['tune'])


