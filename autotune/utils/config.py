# -*- coding: utf-8 -*-

import configparser
import json
from collections import defaultdict

class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d



knob_config = {}

def get_default_dict(dic):
    default_dic =  defaultdict(str)
    for k in dic:
        default_dic[k] = dic[k]

    return default_dic

def parse_args(file):
    cf = DictParser()
    cf.read(file, encoding="utf-8")
    config_dict = cf.read_dict()
    global knob_config
    f = open(config_dict['database']['knob_config_file'])
    knob_config = json.load(f)

    return get_default_dict(config_dict["database"]), get_default_dict(config_dict['tune'])


