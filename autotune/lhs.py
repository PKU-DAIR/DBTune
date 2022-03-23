import os
import bisect
import numpy as np
from pyDOE import lhs
from scipy.stats import uniform


BYTES_SYSTEM = [(1024 ** 5, 'PB'),
                (1024 ** 4, 'TB'),
                (1024 ** 3, 'GB'),
                (1024 ** 2, 'MB'),
                (1024 ** 1, 'KB'),
                (1024 ** 0, 'B'),
]

TIME_SYSTEM = [(1000 * 60 * 60 * 24, 'd'),
               (1000 * 60 * 60, 'h'),
               (1000 * 60, 'min'),
               (1000, 's'),
               (1, 'ms'),
]

class LHSGenerator:

    def __init__(self, sample_num, sample_knobs):
        self.sample_num = sample_num
        self.sample_knobs = sample_knobs

    def __get_raw_size(self, value, system):
        for factor, suffix in system:
            if value.endswith(suffix):
                if len(value) == len(suffix):
                    amount = 1
                else:
                    try:
                        amount = int(value[:-len(suffix)])
                    except ValueError:
                        continue
                return amount * factor
        return None

    def __get_knob_raw(self, value, knob_type):
        if knob_type == 'integer':
            return int(value)
        elif knob_type == 'float':
            return float(value)
        elif knob_type == 'bytes':
            return self.__get_raw_size(value, BYTES_SYSTEM)
        elif knob_type == 'time':
            return self.__get_raw_size(value, TIME_SYSTEM)
        elif knob_type in ['enum', 'combination']:
            return int(value)
        else:
            raise Exception('Knob Type does not support')


    def __get_knob_readable(self, value, knob_type):
        if knob_type == 'integer':
            return int(round(value))
        elif knob_type == 'float':
            return float(value)
        elif knob_type == 'bytes':
            value = int(round(value))
            return size(value, system=BYTES_SYSTEM)
        elif knob_type == 'time':
            value = int(round(value))
            return size(value, system=TIME_SYSTEM)
        elif knob_type in ['enum', 'combination']:
            return int(round(value))
        else:
            raise Exception('Knob Type does not support')

    def __get_knobs_readable(self, values, types):
        result = []
        for i, value in enumerate(values):
            result.append(self.__get_knob_readable(value, types[i]))
        return result

    def generate_results(self):
        names = []
        max_vals = []
        min_vals = []
        types = []
        range_vals = []
        enumerate_vals = {}  # for enumerate type knob only
        combination_vals = {} # for combination type knob only
        for knob, details in self.sample_knobs.items():
            names.append(knob)
            knob_type = details['type']
            if knob_type == 'integer':
                tuning_range = [details['min'], details['max']]
            if knob_type == 'enum':
                tuning_range = [k for k in range(len(details['enum_values']))]
            if knob_type == 'combination':
                tuning_range = [k for k in range(len(details['combination_values']))]
            if knob_type == 'float':
                tuning_range = [details['min'], details['max']]
            max_vals.append(self.__get_knob_raw(tuning_range[-1], knob_type))
            min_vals.append(self.__get_knob_raw(tuning_range[0], knob_type))
            types.append(knob_type)
            if knob_type == 'enum':
                # for enumerate type, the last list denotes the enumerate values
                enumerate_vals[knob] = details['enum_values']
            if knob_type == 'combination':
                combination_vals[knob] = details['combination_values']
            if details.get('stride'):
                stride = float(details['stride'])
                range_vals.append(np.arange(details['min'], details['max']+stride, stride))
            else:
                range_vals.append(None)
        
        nfeats = len(self.sample_knobs)
        samples = lhs(nfeats, samples=self.sample_num, criterion='maximin')
        max_vals = np.array(max_vals)
        min_vals = np.array(min_vals)
        scales = max_vals - min_vals

        for fidx in range(nfeats):
            tmp = uniform(loc=min_vals[fidx], scale=scales[fidx]).ppf(samples[:, fidx])
            if range_vals[fidx] is not None:
                tmp = [range_vals[fidx][bisect.bisect_left(range_vals[fidx], item)] for item in tmp]
            samples[:, fidx] = tmp
        
        samples_readable = []
        for sample in samples:
            samples_readable.append(self.__get_knobs_readable(sample, types))
        
        configs = []
        for sidx in range(self.sample_num):
            config = {}
            for fidx in range(nfeats):
                # handle enumerate type
                if types[fidx] == 'enum':
                    config[names[fidx]] = enumerate_vals[names[fidx]][samples_readable[sidx][fidx]]
                elif types[fidx] == 'combination':
                    kk = names[fidx]
                    vv = combination_vals[names[fidx]][samples_readable[sidx][fidx]]
                    combination_knobs = kk.split('|')
                    combination_values = vv.split('|')
                    for k, knob in enumerate(combination_knobs):
                        config[knob] = combination_values[k]
                else:
                    config[names[fidx]] = samples_readable[sidx][fidx]
            configs.append(config)
        return configs
