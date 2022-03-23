# -*- coding: utf-8 -*-

import configparser
import json


class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


cf = DictParser()
cf.read("config.ini", encoding="utf-8")
config_dict = cf.read_dict()


def parse_args():
    return config_dict["database"], config_dict['tune']


def parse_knob_config():
    f =  open(config_dict['database']['knob_config_file'])
    _knob_config =  json.load(f)
    #for key in _knob_config:
    #    _knob_config[key] = json.loads(str(_knob_config[key]).replace("\'", "\""))
    return _knob_config



knob_config = parse_knob_config()
# sync with main.py


#predictor_output_dim = int(config_dict["predictor"]["predictor_output_dim"])

#predictor_epoch = int(config_dict["predictor"]["predictor_epoch"])
