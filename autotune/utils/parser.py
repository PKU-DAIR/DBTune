import re
import os
import sys
from shutil import copyfile
from dynaconf import settings
import numpy as np
import pandas as pd
import ast
import statistics
from ConfigSpace import Configuration
from ConfigSpace.configuration_space import OrderedDict
from autotune.knobs import logger

TIMEOUT = 4
num_samples_normal = 0

class ConfigParser(object):

    def __init__(self, cnf):
        f = open(cnf)
        self._cnf = cnf
        self._knobs = {}
        for line in f:
            if line.strip().startswith('skip-external-locking') \
                    or line.strip().startswith('[') \
                    or line.strip().startswith('#') \
                    or line.strip() == '':
                pass
            else:
                try:
                    k, _, v = line.strip().split()
                    self._knobs[k] = v
                except:
                    continue
        f.close()

    def replace(self, tmp='/tmp/tmp.cnf'):
        record_list = []
        f1 = open(self._cnf)
        f2 = open(tmp, 'w')
        for line in f1:
            tpl = line.strip().split()
            if len(tpl) < 1:
                f2.write(line)
            elif tpl[0] in self._knobs:
                record_list.append(tpl[0])
                tpl[2] = self._knobs[tpl[0]]
                f2.write('%s\t\t%s %s\n' % (tpl[0], tpl[1], tpl[2]))
            else:
                f2.write(line)
        for key in self._knobs.keys():
            if not key in record_list:
                f2.write('%s\t\t%s %s\n' % (key, '=', self._knobs[key]))
        f1.close()
        f2.close()
        copyfile(tmp, self._cnf)

    def set(self, k, v):
        if isinstance(v, str) and ' ' in v:
            self._knobs[k] = "'{}'".format(v)
        else:
            self._knobs[k] = v


def parse_tpcc(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(".*?trx: (\d+), 95%: (\d+.\d+), 99%: (\d+.\d+), max_rt:.*?")
    temporal = temporal_pattern.findall(lines)
    tps, latency, qps = 0, 0, 0
    for i in temporal:
        tps += float(i[0])
        latency += float(i[2])
    num_samples = len(temporal)
    if num_samples != 0:
        tps /= num_samples
        latency /= num_samples
        # interval
        tps /= 1
    return [tps, latency, tps]


def parse_sysbench(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(
                "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)"
                " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
    temporal = temporal_pattern.findall(lines)
    tps, latency, qps = 0, 0, 0
    tpsL, latL ,qpsL = [], [], []
    for i in temporal:
        tps += float(i[0])
        latency += float(i[5])
        qps += float(i[1])
        tpsL.append(float(i[0]))
        latL.append(float(i[5]))
        qpsL.append(float(i[1]))
    num_samples = len(temporal)
    global num_samples_normal
    if num_samples_normal == 0:
        num_samples_normal = num_samples
    if num_samples >= num_samples_normal * 0.8:
        tps /= num_samples
        qps /= num_samples
        latency /= num_samples
        tps_var = statistics.variance(tpsL)
        lat_var = statistics.variance(latL)
        qps_var = statistics.variance(qpsL)
        return [tps, latency, qps, tps_var, lat_var, qps_var]

    else:
        print('num_samples is zero!')
        return[-1, -1, -1, -1, -1, -1]


def parse_job(file_path, select_file, timeout=4):
    with open(file_path) as f:
        lines = f.readlines()

    with open(select_file) as f:
        lines_select = f.readlines()
    num_sql = len(lines_select)

    latL = []
    for line in lines[1:]:
        if line.strip() == '':
            continue
        tmp = line.split('\t')[-1].strip()
        latL.append(float(tmp)/1000)

    measured_num =  len(latL)
    for i in range(0, num_sql - measured_num):
        latL.append(timeout)

    lat = np.percentile(latL, 95)
    total_time = min(timeout, np.max(latL))
    tps = len(latL) * 60. / total_time

    lat_var = statistics.variance(latL)
    return [tps, lat, tps, -1, lat_var, -1]

def parse_cloudbench(file_path):
    f = open(file_path)
    qps_list = []
    for line in f:
        if 'Request/s' in line:
            v = line.split()[6].split(':')[1]
            qps_list.append(float(v))
    qps = sum(qps_list) / float(len(qps_list))
    return [qps, 0, qps]


def parse_oltpbench(file_path):
    # file_path = *.summary
    with open(file_path) as f:
        lines = f.read()

    tps_temporal_pattern = re.compile("Throughput.*?(\d+.\d+),")
    tps_temporal = tps_temporal_pattern.findall(lines)
    tps = float(tps_temporal[0])

    lat_temporal_pattern = re.compile("95th.*?(\d+.\d+),")
    lat_temporal = lat_temporal_pattern.findall(lines)
    latency = float(lat_temporal[0])

    return [tps, latency, tps, -1, -1, -1]



'''
temporary function: convert internal metrics of 65 dimensions to 51 dimensions
We need this function, because the history internal metrics we collected is 51 dimensions,
and we want to collect all internal metrics(65 dimensions) in later run. So, when mapping,
we convert the dimension for the purpose of matching  history data. 
'''
def convert_65IM_to_51IM(metrics):
    extra_metrics_index = [11, 12, 15, 40, 37, 43, 47, 48, 49, 50, 51, 62, 63, 64]
    if len(metrics.shape) == 2:
        if metrics.shape[1] == 51:
            return metrics
        metrics = np.delete(metrics, extra_metrics_index, axis=1)
    else:
        if metrics.shape[0] == 51:
            return metrics
        metrics = np.delete(metrics, extra_metrics_index)

    return metrics

def get_action_data_from_res_cpu(fn):
    f = open(fn)
    lines = f.readlines()
    f.close()
    metricsL = []
    tpsL, cpuL, internal_metricL = [], [], []
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
        cpu = float(t[4])
        if t[-1] == '65d':
            internal_metric = ast.literal_eval(t[-2])
        else:
            internal_metric = ast.literal_eval(t[-1])

        metricsL.append(valueL)
        tpsL.append(tps)
        cpuL.append(cpu)
        internal_metricL.append(internal_metric)
    df = pd.DataFrame(metricsL, columns=nameL)

    return df, tpsL, cpuL, np.vstack(internal_metricL)


def get_action_data_from_res_cpu2(fn):
    f = open(fn)
    lines = f.readlines()
    f.close()
    metricsL = []
    tpsL, cpuL, latencyL, internal_metricL = [], [], [], []
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
        cpu = float(t[4])
        latency = float(t[2])
        if t[-1] == '65d':
            internal_metric = ast.literal_eval(t[-2])
        else:
            internal_metric = ast.literal_eval(t[-1])

        metricsL.append(valueL)
        tpsL.append(tps)
        cpuL.append(cpu)
        latencyL.append(latency)
        internal_metricL.append(internal_metric)
    df = pd.DataFrame(metricsL, columns=nameL)

    return df, tpsL, cpuL, latencyL, np.vstack(internal_metricL)

def get_action_data_json(file, valid_IM=False):
    f = open(file)
    lines = f.readlines()
    dicL1, dicL2, internal_metricL = [], [], []
    for line in lines:
        if '{' in line:
            line = line[line.index('{'):]
        tmp = line.split('|')
        json_str = tmp[0]
        knob = eval(json_str)
        for key in knob:
            if  not knob[key] == 'ON' and not knob[key] == 'OFF':
                try:
                    knob[key] = int(knob[key])
                except:
                    continue
        metrics = {}
        for i in range(1, len(tmp) - 2):
            key = tmp[i].split('_')[0]
            value = float(tmp[i].split('_')[1])
            metrics[key] = value

        if len(tmp) <= 2:
            internal_metric = [None for i in range(0, 65)]
            if valid_IM:
                continue
        if tmp[-1].strip() == '65d':
                internal_metric = ast.literal_eval(tmp[-2].strip())
                if len(internal_metric) == 0:
                    if valid_IM:
                        continue
                    internal_metric = [None for i in range(0, 65)]
        else:
            internal_metric = ast.literal_eval(tmp[-1].strip())
        dicL1.append(knob)
        dicL2.append(metrics)
        internal_metricL.append(internal_metric)

    df1 = pd.DataFrame.from_dict(dicL1)
    df2 = pd.DataFrame.from_dict(dicL2)

    return df1, df2, np.vstack(internal_metricL)


def get_increment_result(file, default_knobs):
    f = open(file)
    lines = f.readlines()
    dicL1, dicL2, internal_metricL = [], [], []
    for line in lines:
        if '{' in line:
            line = line[line.index('{'):]
        tmp = line.split('|')
        json_str = tmp[0]
        knob = eval(json_str)
        knobs_r = {}
        for key in default_knobs.keys():
            if key not in knob.keys():
                knobs_r[key] = default_knobs[key]
            else:
                knobs_r[key] = knob[key]
            if  not knobs_r[key] == 'ON' and not knobs_r[key] == 'OFF':
                try:
                    knobs_r[key] = int(knobs_r[key])
                except:
                    continue
        metrics = {}
        for i in range(1, len(tmp) - 2):
            key = tmp[i].split('_')[0]
            value = float(tmp[i].split('_')[1])
            metrics[key] = value

        dicL1.append(knobs_r)
        dicL2.append(metrics)


    df1 = pd.DataFrame.from_dict(dicL1)
    df2 = pd.DataFrame.from_dict(dicL2)

    return df1, df2


def get_hist_json(file, cs, y_variable, knob_detail):
    od = OrderedDict()
    f = open(file)
    lines = f.readlines()
    for line in lines:
        if '{' in line:
            line = line[line.index('{'):]
        tmp = line.split('|')
        json_str = tmp[0]
        knob = eval(json_str)
        for key in knob.copy():
            if not key in cs.get_hyperparameter_names():
                knob.pop(key, None)
                logger.info("{key} in {file} is not contained in defined configuration space, removed".format(**locals()))
                continue
            if not knob[key] == 'ON' and not knob[key] == 'OFF':
                try:
                    knob[key] = int(knob[key])
                except:
                    continue
            if knob_detail[key]['max'] > sys.maxsize:
                knob[key] = int(knob[key] / 1000)

        if len(cs.get_hyperparameters()) > len(knob.keys()):
            for key in cs.get_hyperparameters_dict().keys():
                if key not in knob.keys():
                    logger.info("{key} in defined configuration space is not contained in {file}, use default value".format(**locals()))
                    knob[key] = cs.get_hyperparameters_dict()[key].default_value

        config = Configuration(configuration_space=cs, values=knob)

        for i in range(1, len(tmp) - 2):
            key = tmp[i].split('_')[0]
            if key == y_variable:
                perf = float(tmp[i].split('_')[1])

        od[config] = perf

    return od



def is_number(s):
    if s is None:
        return False
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False
