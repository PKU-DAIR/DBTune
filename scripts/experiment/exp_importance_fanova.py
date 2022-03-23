# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np
import re
import sys
import pandas as pd
import pdb
from sklearn import preprocessing
from fanova import fANOVA
import ConfigSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
import math
import fanova.visualizer
output_file="gen_knobs/SYSBENCH_fanova.json"
y_variable = 'tps' #'lat'
regression=True

def get_action_data_json(fn, knobs_template):
    f = open(fn)
    lines = f.readlines()
    f.close()
    metricsL = []
    tpsL, cpuL = [], []
    for line in lines:
        if line[0] == '[':
            continue
        t = line.strip().split('|')
        valueL, nameL = [], []
        knob_str = t[0]
        for key in knobs_template.keys():
            if not key in line:
                continue
            if not knobs_template[key]['type'] == 'enum':
                pattern = r'(' + key + '_([^_]*))'
                value = re.search(pattern, knob_str).groups(0)[1]
            else:
                #pdb.set_trace()
                begin = line.index(key) + len(key)
                min_id = 1e9
                for fkey in knobs_template.keys():
                    id = line[begin:].find(fkey)
                    if id == -1:
                        continue
                    if id < min_id:
                        min_id = id
                value = line[begin: begin + min_id].strip('_')
            nameL.append(key)
            valueL.append(value)

        tps = float(t[1])
        cpu = float(t[3])
        metricsL.append(valueL)
        tpsL.append(tps)
        cpuL.append(cpu)

    df = pd.DataFrame(metricsL, columns=nameL)
    df['tps'] = tpsL
    df['cpu'] = cpuL
    return df


def load_json_res(file, konb_template):
    f = open(file)
    lines = f.readlines()
    dicL = []
    for line in lines:
        if '{' in line:
            line = line[line.index('{'):]
        tmp = line.split('|')
        json_str = tmp[0]
        knob = eval(json_str)
        knob_leave = {}
        for k in konb_template.keys():
            knob_leave[k] = str(knob[k])
        for i in range(1, len(tmp)-2):
            key = tmp[i].split('_')[0]
            value = float(tmp[i].split('_')[1])
            knob_leave[key] = value
        dicL.append(knob_leave)
    df = pd.DataFrame.from_dict(dicL)

    return df

def plot_dis(df):
    plt.figure()
    sns.set_style("ticks")
    '''sns.distplot(df, hist=False, kde=True, rug=True,  # 选择是否显示条形图、密度曲线、观测的小细条（边际毛毯）
                 kde_kws={"color": "lightcoral", "lw": 1.5, 'linestyle': '--'},  # 设置选择True的条件(其密度曲线颜色、线宽、线形)
                 rug_kws={'color': 'lightcoral', 'alpha': 1, 'lw': 2, }, label='TPS')'''
    sns.boxplot(y=df, label='TPS')
    plt.savefig('plot/tps.png')

if __name__ == '__main__':
    f_json = sys.argv[1]
    fn = sys.argv[2]
    f_json = open(f_json)
    konb_template = json.load(f_json)
    #konb_template = json.loads(konb_template)
    #action_df = get_action_data_json(fn, konb_template)
    action_df = load_json_res(fn, konb_template)
    #action_dConfigSpace.f = action_df.drop('large_pages',axis=1)
    if regression:

        action_df = action_df.dropna(axis=0, how='any', subset=['tps'])
        # action_df.loc[action_df['tps'] != -1, 'tps'] = 0
        action_df = action_df[~action_df['tps'].isin([-1, -2, -3, -4])]  # 去掉sysbench不成功的样本
        #action_df = action_df[action_df['completion_type']=='NO_CHAIN']
        action_df = action_df[action_df['tps'] > 0]
        action_df = action_df[action_df['tps'] != action_df['tps'].max()]
        #action_df = action_df[action_df['tps'] > 300]
        X = action_df.iloc[:, :-12]
        cs = ConfigSpace.ConfigurationSpace()
        le = preprocessing.LabelEncoder()
        for c in list(X.columns)[:-2]:
            if konb_template[c]['type'] == 'enum':
                le.fit(X[c])
                X[c] = le.transform(X[c])
                default_transformed = le.transform(np.array(str(konb_template[c]['default'])).reshape(1))[0]
                list_transformed = X[c].unique().tolist()
                knob = CategoricalHyperparameter(c, list_transformed, default_value=default_transformed)
            else:
                X[c] = X[c].astype('float')
                if konb_template[c]['min'] > X[c].min():
                    X = X[X[c] > konb_template[c]['min']]
                    print (c)
                if konb_template[c]['max'] < X[c].max():
                    X = X[X[c] < konb_template[c]['max']]
                    print (c)
                if konb_template[c]['max'] > 2**20:
                    #X = X.drop([c], axis=1)
                    #continue
                    #X[c] = (X[c]/(2**20))
                    #knob = UniformFloatHyperparameter(c, konb_template[c]['min']/(2**20),  konb_template[c]['max']/(2**20), default_value=konb_template[c]['default']/(2**20))
                    X[c] = (X[c] -konb_template[c]['min'] )/konb_template[c]['max']
                    knob = UniformFloatHyperparameter(c, 0, 1, default_value=(konb_template[c]['default'] -konb_template[c]['min'])/ konb_template[c]['max'])
                else:
                    knob = UniformIntegerHyperparameter(c, konb_template[c]['min'], konb_template[c]['max'], default_value=konb_template[c]['default'])


            cs.add_hyperparameter(knob)

        Y = X[y_variable].astype('float')
        X = X.drop(['tps', 'lat'], axis=1)
        #X = X.iloc[:1000,:]
        X.index = range(0, X.shape[0])
        Y.index = range(0, Y.shape[0])
        #Y.index = range(0, 1000)
        #X = np.array(X)
        #Y = np.array(Y)
        f = fANOVA(X, Y, config_space=cs)
        im_dir = {}
        for i in list(X.columns):
            value = f.quantify_importance((i,))[(i,)]['individual importance']
            if not math.isnan(value):
                im_dir[i] = value

        a = sorted(im_dir.items(), key=lambda x: x[1], reverse=True)
        print('Knobs importance rank:')
        out_knob = {}
        for i in range(0, len(a)):
            if a[i][1] != 0:
                print("Top{}: {}, its importance accounts for {:.4%}".format(i + 1, a[i][0],
                                                                               a[i][1] ))
                knob = a[i][0]
                konb_template[knob]['important_rank'] = i + 1
                out_knob[knob] = konb_template[knob]



        for key in konb_template.keys():
            if key not in out_knob.keys():
                konb_template[key]['important_rank'] = -1
                out_knob[key] = konb_template[key]

        with open(output_file, 'w') as fp:
            json.dump(out_knob, fp, indent=4)


        #vis = fanova.visualizer.Visualizer(f, cs, "./plot")
        pdb.set_trace()
        #vis.create_all_plots()
        # action_df = action_df[action_df['tps']>100]
        # pdb.set_trace()
        # pdb.set_trace()
        # action_df = action_df0[actipon_df0['tps'] > int(C) ]
        # action_df = action_df1[action_df1['tps'] > 900]
